"""按引擎准备「基础权重」。

Ultralytics 的权重由项目自己的下载器负责；RF-DETR 与独立版 RT-DETR 的权重来自各自框架，
这里把它们统一成同一个状态机，供接口与界面复用：

- `status()`  只查询，不触发下载
- `start()`   后台线程下载，立即返回当前状态
"""

from __future__ import annotations

import threading
from pathlib import Path

from app.core.errors import ValidationError

PREP_ENGINES = ("rfdetr", "rtdetrv2")

# RF-DETR 已发布的档位 -> 官方权重文件名
RF_VARIANT_FILES = {
    "nano": "rf-detr-nano.pth",
    "small": "rf-detr-small.pth",
    "base": "rf-detr-base.pth",
    "medium": "rf-detr-medium.pth",
    "large": "rf-detr-large.pth",
}
RF_DEFAULT_VARIANT = "base"

_LOCK = threading.Lock()
_STATE: dict[str, dict] = {}


def supports(engine_key: str) -> bool:
    return (engine_key or "").strip().lower() in PREP_ENGINES


def _state_for(key: str) -> dict:
    with _LOCK:
        state = dict(_STATE.get(key) or {})
    state.setdefault("status", "idle")
    state.setdefault("percent", 0.0)
    state.setdefault("error", None)
    return state


def _set_state(key: str, **kwargs) -> None:
    with _LOCK:
        state = dict(_STATE.get(key) or {})
        state.update(kwargs)
        _STATE[key] = state


def _rf_target(name: str) -> tuple[Path, str]:
    """返回 RF-DETR 权重的目标路径与文件名。"""
    from app.engines.rfdetr_home import ensure_rf_home

    # 先把缓存目录指到项目内，避免用户目录不可写时直接失败
    ensure_rf_home()
    from rfdetr.assets.model_weights import get_model_cache_dir

    variant = (name or "").strip().lower() or RF_DEFAULT_VARIANT
    filename = RF_VARIANT_FILES.get(variant)
    if filename is None:
        raise ValidationError(
            "weight_variant_unknown",
            f"未知的 RF-DETR 档位：{name}",
            details={"available": sorted(RF_VARIANT_FILES)},
        )
    return Path(get_model_cache_dir()) / filename, filename


def _rtdetrv2_target(name: str) -> tuple[Path, str]:
    from app.engines.rtdetrv2_weights import VARIANT_DEPTH, backbone_path

    variant = (name or "").strip().lower() or "r18"
    depth = VARIANT_DEPTH.get(variant)
    if depth is None:
        raise ValidationError(
            "weight_variant_unknown",
            f"未知的 RT-DETR（独立版）档位：{name}",
            details={"available": sorted(VARIANT_DEPTH)},
        )
    path = backbone_path(depth)
    return path, path.name


def _target(engine_key: str, name: str) -> tuple[Path, str]:
    key = (engine_key or "").strip().lower()
    if key == "rfdetr":
        return _rf_target(name)
    if key == "rtdetrv2":
        return _rtdetrv2_target(name)
    raise ValidationError("weight_prep_unsupported", f"引擎 {engine_key} 不支持预置权重。")


def status(engine_key: str, name: str) -> dict:
    path, filename = _target(engine_key, name)
    ready = path.is_file() and path.stat().st_size > 0
    return {
        "name": filename,
        "ready": ready,
        "kind": "engine",
        "path": str(path),
        "size_bytes": path.stat().st_size if ready else None,
        "download": _state_for(f"{engine_key}:{filename}"),
    }


def _download_rfdetr(name: str) -> Path:
    """并行下载 RF-DETR 官方权重；框架自带的单连接在国内太慢。"""
    from rfdetr.assets.model_weights import ModelWeights

    from app.engines import rfdetr_download

    path, filename = _rf_target(name)
    url = ModelWeights.get_url(filename)
    if not url:
        raise ValidationError("weight_url_missing", f"找不到 {filename} 的下载地址")
    state_key = f"rfdetr:{filename}"

    def on_progress(done: int, total: int) -> None:
        percent = round(done * 100.0 / total, 1) if total else 0.0
        _set_state(state_key, downloaded=done, total=total, percent=percent)

    result = rfdetr_download.download(url, path, on_progress=on_progress)
    expected = ModelWeights.get_md5(filename)
    if expected:
        import hashlib

        digest = hashlib.md5()
        with open(result, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            result.unlink(missing_ok=True)
            raise IOError("权重校验失败，文件可能不完整，请重试")
    return result


def _worker(engine_key: str, name: str) -> None:
    path, filename = _target(engine_key, name)
    key = f"{engine_key}:{filename}"
    try:
        if engine_key == "rfdetr":
            _download_rfdetr(name)
        else:
            from app.engines.rtdetrv2_weights import VARIANT_DEPTH, ensure_backbone

            ensure_backbone(VARIANT_DEPTH[(name or "r18").strip().lower() or "r18"])
        _set_state(key, status="ready", percent=100.0, error=None)
    except Exception as exc:  # 网络/磁盘问题都记录下来给界面
        _set_state(key, status="failed", error=f"{type(exc).__name__}: {exc}")


def start(engine_key: str, name: str) -> dict:
    path, filename = _target(engine_key, name)
    key = f"{engine_key}:{filename}"
    current = _state_for(key)
    if path.is_file() and path.stat().st_size > 0:
        _set_state(key, status="ready", percent=100.0, error=None)
        return status(engine_key, name)
    if current["status"] == "downloading":
        return status(engine_key, name)
    _set_state(key, status="downloading", percent=0.0, error=None)
    threading.Thread(target=_worker, args=(engine_key, name), daemon=True).start()
    return status(engine_key, name)
