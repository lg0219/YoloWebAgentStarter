from __future__ import annotations

"""智能分析（应用态）代理：把 vlm-bridge 的能力挂到 YWA 的 /api 下。

设计：YWA 只做入口与台账转发，推理全部在 vlm-bridge（127.0.0.1:8002）。
好处：训练/标注不受推理拖累；任一服务重启互不影响。
"""
import json
import os
from pathlib import Path

import httpx
from fastapi import APIRouter, Body, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse, Response

from app.api.dependencies import get_settings, get_storage
from app.core.config import Settings
from app.core.errors import NotFoundError, ValidationError
from app.core.storage import Storage

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

BRIDGE_URL = os.environ.get("YWA_BRIDGE_URL", "http://127.0.0.1:8002")
# 默认与 vlm-bridge 服务自身的工作目录一致（同机部署零配置）；
# 跨机部署时用 VLM_JOBS_DIR 指向本机可读的同一份目录。
_REPO_ROOT = Path(__file__).resolve().parents[4]
BRIDGE_JOBS_DIR = Path(os.environ.get("VLM_JOBS_DIR", str(_REPO_ROOT / "services" / "vlm-bridge" / "jobs")))
PROXY_TIMEOUT = float(os.environ.get("YWA_BRIDGE_TIMEOUT", "120"))


def _cloud_override(settings: Settings) -> dict | None:
    """读取设置页里的云端解释配置，作为单次参数传给 vlm-bridge（不落盘到配置文件）。"""
    from app.settings.service import SettingsService

    return SettingsService(settings).pipeline_cloud_override()


def _proxy(method: str, path: str, **kwargs) -> Response:
    url = BRIDGE_URL.rstrip("/") + path
    try:
        with httpx.Client(timeout=PROXY_TIMEOUT) as client:
            upstream = client.request(method, url, **kwargs)
    except httpx.HTTPError as exc:
        raise ValidationError("bridge_unavailable", f"推理服务不可用：{exc}") from exc
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
        headers={"content-disposition": upstream.headers["content-disposition"]}
        if "content-disposition" in upstream.headers
        else None,
    )


@router.get("/health")
def bridge_health() -> Response:
    return _proxy("GET", "/health")


@router.get("/templates")
def bridge_templates() -> Response:
    return _proxy("GET", "/templates")


@router.get("/jobs")
def jobs() -> Response:
    return _proxy("GET", "/batch")


def _allowed_roots(settings: Settings) -> list[Path]:
    roots = [settings.data_dir, settings.import_root, BRIDGE_JOBS_DIR]
    for extra in os.environ.get("YWA_PIPELINE_ROOTS", "").split(os.pathsep):
        if extra.strip():
            roots.append(Path(extra.strip()))
    resolved: list[Path] = []
    for root in roots:
        try:
            candidate = root.expanduser().resolve()
        except OSError:
            continue
        if candidate.is_dir():
            resolved.append(candidate)
    return resolved


def _check_directory(raw: str, settings: Settings) -> Path:
    path = Path(raw).expanduser()
    try:
        path = path.resolve()
    except OSError as exc:
        raise ValidationError("directory_invalid", f"目录无法解析：{raw}") from exc
    if not path.is_dir():
        raise ValidationError("directory_missing", f"目录不存在：{raw}")
    for root in _allowed_roots(settings):
        if path == root or root in path.parents:
            return path
    raise ValidationError("directory_not_allowed", f"目录不在允许范围：{raw}（可在 YWA_PIPELINE_ROOTS 中追加根目录）")


@router.get("/roots")
def allowed_roots(settings: Settings = Depends(get_settings)) -> dict:
    return {"roots": [str(root) for root in _allowed_roots(settings)]}


@router.post("/batch")
def create_batch(
    payload: dict = Body(...),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> Response:
    """建批：source=dir（生产目录）/ dataset（回归测试）。"""
    source = str(payload.get("source") or "dir")
    if source == "dataset":
        dataset_id = str(payload.get("dataset_id") or "").strip()
        if not dataset_id:
            raise ValidationError("dataset_required", "缺少 dataset_id")
        images_dir = storage.images_dir(dataset_id)
        if not images_dir.exists():
            raise NotFoundError("dataset_dir_missing", f"数据集图片目录不存在：{dataset_id}")
        target_dir = images_dir
        title = payload.get("title") or f"回归-{dataset_id}"
    else:
        raw_dir = str(payload.get("dir") or "").strip()
        if not raw_dir:
            raise ValidationError("directory_required", "缺少目录路径")
        target_dir = _check_directory(raw_dir, settings)
        title = payload.get("title") or target_dir.name or str(target_dir)
    body = {
        "template": payload.get("template"),
        "dir": str(target_dir),
        "limit": int(payload.get("limit") or 0),
        "save_images": bool(payload.get("save_images")),
        "recursive": bool(payload.get("recursive")),
        "skip_processed": bool(payload.get("skip_processed")),
        "title": title,
        "cloud": _cloud_override(settings),
    }
    return _proxy("POST", "/batch", json=body)


@router.post("/batch/upload")
async def upload_batch(
    template: str = Form(""),
    save_images: bool = Form(False),
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
) -> Response:
    """浏览器直传少量图片建批（几百张以内）。"""
    payload_files = []
    for upload in files:
        content = await upload.read()
        payload_files.append(
            ("files", (upload.filename or "image.jpg", content, upload.content_type or "image/jpeg"))
        )
    form = {"template": template, "save_images": "true" if save_images else "false"}
    override = _cloud_override(settings)
    if override:
        form["cloud"] = json.dumps(override, ensure_ascii=False)
    with httpx.Client(timeout=600) as client:
        upstream = client.post(
            BRIDGE_URL.rstrip("/") + "/batch/upload",
            data=form,
            files=payload_files,
        )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )


@router.get("/batch/{job_id}")
def batch_status(job_id: str) -> Response:
    return _proxy("GET", f"/batch/{job_id}")


@router.get("/batch/{job_id}/image/{name}")
def batch_original_image(
    job_id: str,
    name: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """按批次明细里的文件名返回原图，供页面点击放大。"""
    for item in _read_items(job_id):
        if item.get("name") != name:
            continue
        candidate = Path(item.get("path") or "")
        if not candidate.is_file():
            raise NotFoundError("image_missing", "原图不存在")
        resolved = candidate.resolve()
        for root in _allowed_roots(settings):
            if resolved == root or root in resolved.parents:
                return FileResponse(resolved)
        raise ValidationError("path_not_allowed", "图片不在允许目录内")
    raise NotFoundError("item_not_found", "批次中没有该图片")


@router.get("/batch/{job_id}/items")
def batch_items(job_id: str, offset: int = 0, limit: int = 200) -> Response:
    return _proxy("GET", f"/batch/{job_id}/items", params={"offset": offset, "limit": limit})


@router.get("/batch/{job_id}/export.csv")
def batch_csv(job_id: str) -> Response:
    return _proxy("GET", f"/batch/{job_id}/export.csv")


@router.post("/batch/{job_id}/cancel")
def batch_cancel(job_id: str) -> Response:
    return _proxy("POST", f"/batch/{job_id}/cancel")


@router.get("/batch/{job_id}/annotated/{name}")
def annotated(job_id: str, name: str) -> FileResponse:
    base = (BRIDGE_JOBS_DIR / job_id / "annotated").resolve()
    target = (base / name).resolve()
    if base not in target.parents and target != base:
        raise ValidationError("path_invalid", "非法文件路径")
    if not target.is_file():
        raise NotFoundError("annotated_missing", "标注图不存在")
    return FileResponse(target)



# ---------- 第三步：闭环（回灌待标池 / 日报） ----------


def _read_meta(job_id: str) -> dict | None:
    path = BRIDGE_JOBS_DIR / job_id / "meta.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_items(job_id: str) -> list[dict]:
    path = BRIDGE_JOBS_DIR / job_id / "items.jsonl"
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@router.get("/batch/{job_id}/report")
def batch_report(job_id: str) -> dict:
    """生成该批次的生产日报（Markdown + 结构化统计）。"""
    meta = _read_meta(job_id)
    if meta is None:
        raise NotFoundError("job_not_found", f"批次不存在：{job_id}")
    items = _read_items(job_id)
    done = [item for item in items if item.get("ok")]
    failed = [item for item in items if not item.get("ok")]
    detected = [item for item in done if int((item.get("summary") or {}).get("count") or 0) > 0]
    confs = [float((item.get("summary") or {}).get("max_conf") or 0.0) for item in done]
    durations: list[int] = []
    for item in done:
        total = 0
        for key, value in (item.get("timings") or {}).items():
            if key.endswith("_ms") and isinstance(value, int):
                total += value
        if total:
            durations.append(total)

    detect_rate = (len(detected) / len(done) * 100) if done else 0.0
    avg_ms = (sum(durations) / len(durations)) if durations else 0.0
    stats = {
        "template": meta.get("template"),
        "status": meta.get("status"),
        "created_at": meta.get("created_at"),
        "total": meta.get("total", 0),
        "processed": len(items),
        "ok": len(done),
        "failed": len(failed),
        "detected": len(detected),
        "detect_rate": round(detect_rate, 2),
        "avg_ms": round(avg_ms, 1),
        "max_conf": round(max(confs) if confs else 0.0, 4),
        "elapsed_sec": meta.get("elapsed_sec"),
    }

    top = sorted(detected, key=lambda item: float((item.get("summary") or {}).get("max_conf") or 0), reverse=True)[:10]
    lines = [
        f"# 智能分析日报 · {job_id}",
        "",
        f"- 模板：{stats['template']}",
        f"- 开始时间：{stats['created_at']}",
        f"- 处理张数：{stats['processed']}/{stats['total']}，失败 {stats['failed']}",
        f"- 异常张数：{stats['detected']}，异常率 {stats['detect_rate']}%",
        f"- 平均耗时：{stats['avg_ms']} ms/张",
        f"- 批次状态：{stats['status']}，总耗时 {stats['elapsed_sec']}s",
        "",
        "## 异常明细（按置信度前 10）",
        "",
        "| # | 文件 | 置信度 | 面积占比 | 结论 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, item in enumerate(top, start=1):
        summary = item.get("summary") or {}
        text = ((item.get("reasoning") or {}).get("text") or "").replace("|", "/").replace("\n", " ")
        ratio = summary.get("mask_area_ratio_sum")
        ratio_text = f"{ratio * 100:.2f}%" if isinstance(ratio, (int, float)) else "-"
        lines.append(
            f"| {index} | {item.get('name')} | {summary.get('max_conf')} | {ratio_text} | {text} |"
        )
    return {"job_id": job_id, "stats": stats, "markdown": "\n".join(lines)}


VIEWER_FILE = Path(__file__).with_name("pipeline_viewer.html")


@router.get("/viewer")
def viewer() -> FileResponse:
    """原图查看器：支持缩放/拖拽，叠加检测框。"""
    return FileResponse(VIEWER_FILE, media_type="text/html; charset=utf-8")

