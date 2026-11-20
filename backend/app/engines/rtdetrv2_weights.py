"""独立版 RT-DETR（v2）的预训练权重。

每个结构档位都有官方在 COCO 上训练好的完整权重，界面里选档位就是选它。
应用层直接用预训练起点，不从零训练；下载走项目统一的并行下载器（带百分比）。
"""

from __future__ import annotations

import os
from pathlib import Path

from app.core.config import Settings

# 档位 -> 官方完整 COCO 权重（实测均可下载，大小为 2026-11 实测值）
FULL_WEIGHTS = {
    "r18": "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
    "r34": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth",
    "r50": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth",
    "r101": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth",
}

# 官方下载走 GitHub，国内直连不稳，沿用项目镜像策略
MIRROR_PREFIXES = ("https://ghfast.top", "https://gh-proxy.com")


def weights_dir() -> Path:
    override = os.getenv("YWA_RTDETRV2_WEIGHTS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Settings.from_env().project_root / "weights" / "rtdetrv2"


def weight_path(variant: str) -> Path:
    key = (variant or "r18").strip().lower()
    url = FULL_WEIGHTS.get(key, FULL_WEIGHTS["r18"])
    return weights_dir() / Path(url).name


def candidate_urls(variant: str) -> list[str]:
    key = (variant or "r18").strip().lower()
    official = FULL_WEIGHTS.get(key, FULL_WEIGHTS["r18"])
    return [f"{prefix}/{official}" for prefix in MIRROR_PREFIXES] + [official]


def ensure_weight(variant: str, on_progress=None) -> Path:
    """确保该档位的完整 COCO 权重就位；已存在直接返回。"""
    from app.engines import engine_download

    key = (variant or "r18").strip().lower()
    if key not in FULL_WEIGHTS:
        key = "r18"
    target = weight_path(key)
    if target.is_file() and target.stat().st_size > 0:
        if on_progress is not None:
            on_progress(target.stat().st_size, target.stat().st_size)
        return target

    last_error: Exception | None = None
    for url in candidate_urls(key):
        try:
            return engine_download.download(url, target, on_progress=on_progress)
        except Exception as exc:  # 换下一个通道
            last_error = exc
    raise RuntimeError(f"预训练权重下载失败：{last_error}") from last_error
