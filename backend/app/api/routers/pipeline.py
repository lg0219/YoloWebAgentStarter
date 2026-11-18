from __future__ import annotations

"""智能分析（应用态）代理：把 vlm-bridge 的能力挂到 YWA 的 /api 下。

设计：YWA 只做入口与台账转发，推理全部在 vlm-bridge（127.0.0.1:8002）。
好处：训练/标注不受推理拖累；任一服务重启互不影响。
"""
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
