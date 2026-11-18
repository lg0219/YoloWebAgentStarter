from __future__ import annotations

"""智能分析（应用态）代理：把 vlm-bridge 的能力挂到 YWA 的 /api 下。

设计：YWA 只做入口与台账转发，推理全部在 vlm-bridge（127.0.0.1:8002）。
好处：训练/标注不受推理拖累；任一服务重启互不影响。
"""
import json
import os
from pathlib import Path

import httpx
from fastapi import APIRouter, Body, Depends
from fastapi.responses import FileResponse, Response

from app.api.dependencies import get_storage
from app.core.errors import NotFoundError, ValidationError
from app.core.storage import Storage

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

BRIDGE_URL = os.environ.get("YWA_BRIDGE_URL", "http://127.0.0.1:8002")
BRIDGE_JOBS_DIR = Path(os.environ.get("VLM_JOBS_DIR", "/home/linux/vlm-bridge/jobs"))
PROXY_TIMEOUT = float(os.environ.get("YWA_BRIDGE_TIMEOUT", "120"))


def _proxy(method: str, path: str, **kwargs) -> Response:
    url = BRIDGE_URL.rstrip("/") + path
    try:
        with httpx.Client(timeout=PROXY_TIMEOUT) as client:
            upstream = client.request(method, url, **kwargs)
    except httpx.HTTPError as exc:
        raise ValidationError(f"推理服务不可用：{exc}") from exc
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


@router.post("/batch")
def create_batch(
    payload: dict = Body(...),
    storage: Storage = Depends(get_storage),
) -> Response:
    dataset_id = str(payload.get("dataset_id") or "").strip()
    if not dataset_id:
        raise ValidationError("缺少 dataset_id")
    images_dir = storage.images_dir(dataset_id)
    if not images_dir.exists():
        raise NotFoundError(f"数据集图片目录不存在：{dataset_id}")
    body = {
        "template": payload.get("template"),
        "dir": str(images_dir),
        "limit": int(payload.get("limit") or 0),
        "save_images": bool(payload.get("save_images")),
        "title": payload.get("title") or dataset_id,
    }
    return _proxy("POST", "/batch", json=body)


@router.get("/batch/{job_id}")
def batch_status(job_id: str) -> Response:
    return _proxy("GET", f"/batch/{job_id}")


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
        raise ValidationError("非法文件路径")
    if not target.is_file():
        raise NotFoundError("标注图不存在")
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


def _chunks(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


@router.post("/batch/{job_id}/promote")
def promote_batch(job_id: str, payload: dict | None = Body(default=None)) -> dict:
    """把批次里符合条件的图片回灌成一个新数据集（待复标池）。"""
    options = payload or {}
    if _read_meta(job_id) is None:
        raise NotFoundError(f"批次不存在：{job_id}")
    items = _read_items(job_id)
    if not items:
        raise ValidationError("该批次还没有结果")

    only_detected = bool(options.get("only_detected", True))
    min_conf = float(options.get("min_conf") or 0.0)
    target = options.get("dataset_id")
    picked: list[str] = []
    for item in items:
        if not item.get("ok"):
            continue
        summary = item.get("summary") or {}
        if only_detected and int(summary.get("count") or 0) <= 0:
            continue
        if float(summary.get("max_conf") or 0.0) < min_conf:
            continue
        candidate = item.get("path")
        if candidate and Path(candidate).is_file():
            picked.append(candidate)
    if not picked:
        raise ValidationError("没有符合条件的图片：可放宽置信度，或取消“仅检出”")

    base = f"http://127.0.0.1:{os.environ.get('YWA_PORT', '8001')}"
    imported = 0
    with httpx.Client(timeout=600) as client:
        if not target:
            name = str(options.get("name") or f"待复标-{job_id}")[:200]
            created = client.post(
                base + "/api/datasets",
                json={
                    "name": name,
                    "task_type": "detect",
                    "description": f"来自智能分析批次 {job_id}；待复核 {len(picked)} 张",
                },
            )
            if created.status_code >= 400:
                raise ValidationError(f"创建数据集失败：{created.text[:200]}")
            target = created.json()["id"]
        for chunk in _chunks(picked, 40):
            handles: list = []
            files: list = []
            try:
                for path in chunk:
                    handle = open(path, "rb")
                    handles.append(handle)
                    files.append(("files", (Path(path).name, handle, "image/jpeg")))
                uploaded = client.post(
                    base + f"/api/datasets/{target}/images/upload",
                    files=files,
                    data={"split": "train"},
                )
            finally:
                for handle in handles:
                    handle.close()
            if uploaded.status_code >= 400:
                raise ValidationError(f"导入失败：{uploaded.text[:200]}")
            imported += int(uploaded.json().get("imported") or 0)
    return {"job_id": job_id, "dataset_id": target, "picked": len(picked), "imported": imported}


@router.get("/batch/{job_id}/report")
def batch_report(job_id: str) -> dict:
    """生成该批次的生产日报（Markdown + 结构化统计）。"""
    meta = _read_meta(job_id)
    if meta is None:
        raise NotFoundError(f"批次不存在：{job_id}")
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
