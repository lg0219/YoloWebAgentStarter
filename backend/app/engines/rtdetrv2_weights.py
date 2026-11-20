"""独立版 RT-DETR（v2）的主干预训练权重。

上游代码默认从 github.com 直连下载；本模块沿用项目已有的镜像策略，先试镜像再回退官方，
并把文件缓存到权重目录，供子进程按本地路径加载。
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from app.core.config import Settings
from app.training.weights import MIRRORS

# 与内嵌源码 `src/nn/backbone/presnet.py` 中的 donwload_url 保持一致
BACKBONE_URLS = {
    18: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth",
    34: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth",
    50: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth",
    101: "https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth",
}

# 档位 -> 主干深度
VARIANT_DEPTH = {"r18": 18, "r34": 34, "r50": 50, "r101": 101}


def backbone_dir() -> Path:
    override = os.getenv("YWA_RTDETRV2_WEIGHTS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Settings.from_env().project_root / "weights" / "rtdetrv2"


def backbone_path(depth: int) -> Path:
    return backbone_dir() / Path(BACKBONE_URLS[depth]).name


def candidate_urls(depth: int) -> list[str]:
    official = BACKBONE_URLS[depth]
    urls = [f"{prefix}/{official}" for prefix in MIRRORS.values()]
    urls.append(official)
    return urls


def ensure_backbone(depth: int, log=print) -> Path:
    """确保主干预训练权重就位；已存在则直接返回。"""
    if depth not in BACKBONE_URLS:
        raise ValueError(f"unsupported backbone depth: {depth}")
    target = backbone_path(depth)
    if target.is_file() and target.stat().st_size > 0:
        log(f"[rtdetrv2] backbone ready: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in candidate_urls(depth):
        temporary = target.with_suffix(target.suffix + ".part")
        try:
            log(f"[rtdetrv2] downloading backbone: {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "YWA/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if temporary.stat().st_size <= 0:
                raise IOError("empty download")
            temporary.replace(target)
            log(f"[rtdetrv2] backbone saved: {target} ({target.stat().st_size} bytes)")
            return target
        except Exception as exc:  # 换下一个通道
            last_error = exc
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"预训练权重下载失败：{last_error}") from last_error
