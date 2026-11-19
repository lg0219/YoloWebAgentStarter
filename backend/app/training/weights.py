"""（lg0219）预训练权重管理：状态检查与手动下载。

设计：训练前必须先完成权重下载，训练命令不再触发 Ultralytics 的下载，
因此镜像只在本模块生效，且下载走 Python 而非 curl（避免 Windows Schannel 问题）。
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from pathlib import Path

from app.core.config import Settings
from app.core.errors import ValidationError

# 镜像只是加速通道，文件仍来自官方仓库
MIRRORS = {
    "ghfast": "https://ghfast.top",
    "gh-proxy": "https://gh-proxy.com",
    "ghproxy-net": "https://ghproxy.net",
}
# 与当前 Ultralytics 版本使用的 release 标签保持一致（yolo26/yolo11 等在此标签下）
OFFICIAL_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

_STATE_LOCK = threading.Lock()
_STATE: dict[str, dict] = {}


def _settings_json(settings: Settings) -> dict:
    try:
        path = settings.data_dir / "settings.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def weights_dir(settings: Settings) -> Path:
    """权重目录：优先采用 Ultralytics 自身设置，保证训练时能本地命中。"""
    config_dir = os.environ.get("YOLO_CONFIG_DIR", "").strip()
    candidates = []
    if config_dir:
        candidates.append(Path(config_dir) / "Ultralytics" / "settings.json")
    candidates.append(settings.data_dir / "ultralytics" / "Ultralytics" / "settings.json")
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            value = str(data.get("weights_dir") or "").strip()
            if value:
                return Path(value).expanduser()
        except Exception:
            continue
    return settings.project_root / "weights"


def source_prefix(settings: Settings) -> str:
    runtime = _settings_json(settings).get("runtime") or {}
    source = str(runtime.get("model_source") or "official").strip()
    if source == "custom":
        return str(runtime.get("model_source_custom") or "").strip().rstrip("/")
    return MIRRORS.get(source, "").rstrip("/")


def download_url(name: str, settings: Settings) -> str:
    prefix = source_prefix(settings)
    base = f"{OFFICIAL_BASE}/{name}"
    return f"{prefix}/{base}" if prefix else base


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def check_weight(name: str, settings: Settings) -> dict:
    """判断权重是否已就绪；URL、绝对路径、weights 目录中的文件都算就绪。"""
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValidationError("weight_name_missing", "请先选择基础权重。")
    if _is_url(cleaned):
        return {"name": cleaned, "ready": True, "kind": "url", "path": cleaned, "size_bytes": None}
    candidate = Path(cleaned).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
        return {
            "name": cleaned,
            "ready": resolved.is_file(),
            "kind": "file",
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
        }
    target = weights_dir(settings) / cleaned
    ready = target.is_file()
    return {
        "name": cleaned,
        "ready": ready,
        "kind": "asset",
        "path": str(target),
        "size_bytes": target.stat().st_size if ready else None,
    }


def download_state(name: str) -> dict:
    with _STATE_LOCK:
        return dict(_STATE.get(name) or {"status": "idle", "downloaded": 0, "total": 0, "percent": 0.0, "error": None})


def _set_state(name: str, **kwargs) -> None:
    with _STATE_LOCK:
        state = dict(_STATE.get(name) or {})
        state.update(kwargs)
        _STATE[name] = state


def start_download(name: str, settings: Settings) -> dict:
    """启动后台下载；已在下载中则直接返回当前状态。"""
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValidationError("weight_name_missing", "请先选择基础权重。")
    if _is_url(cleaned) or Path(cleaned).parent != Path("."):
        raise ValidationError("weight_not_downloadable", "该权重不是官方资产名，无法自动下载。")
    if not cleaned.endswith(".pt"):
        raise ValidationError("weight_not_downloadable", "只有 .pt 权重需要下载。")

    current = download_state(cleaned)
    if current["status"] == "downloading":
        return current

    target = weights_dir(settings) / cleaned
    if target.is_file():
        _set_state(cleaned, status="ready", percent=100.0, error=None,
                   total=target.stat().st_size, downloaded=target.stat().st_size)
        return download_state(cleaned)

    url = download_url(cleaned, settings)
    _set_state(cleaned, status="downloading", downloaded=0, total=0, percent=0.0, error=None, url=url)
    thread = threading.Thread(target=_download_worker, args=(cleaned, url, target), daemon=True)
    thread.start()
    return download_state(cleaned)


def _download_worker(name: str, url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "YWA/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("content-length") or 0)
            _set_state(name, total=total)
            downloaded = 0
            with open(temporary, "wb") as handle:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    percent = round(downloaded * 100.0 / total, 1) if total else 0.0
                    _set_state(name, downloaded=downloaded, percent=percent)
        if total and temporary.stat().st_size < total:
            raise IOError("下载不完整")
        temporary.replace(target)
        _set_state(name, status="ready", percent=100.0, error=None)
    except Exception as exc:  # 网络/磁盘错误都按失败处理，保留原因供前端提示
        _set_state(name, status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def require_ready(name: str, settings: Settings) -> None:
    """训练前校验：权重未就绪直接拒绝。"""
    info = check_weight(name, settings)
    if not info["ready"]:
        raise ValidationError(
            "weight_not_downloaded",
            "模型权重未下载，请先下载！",
            details={"name": info["name"]},
        )
