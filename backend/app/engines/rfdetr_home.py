"""RF-DETR 的权重缓存目录。

框架默认把权重放进 `~/.roboflow/models`。在受限账户或受限执行环境下，用户目录不可写，
会直接抛 `PermissionError [WinError 5]`。这里统一改到项目内的可写目录，用户显式设置
`RF_HOME` 时仍然尊重用户的选择。
"""

from __future__ import annotations

import os
from pathlib import Path

from app.core.config import Settings

ENV_KEY = "RF_HOME"


def default_home() -> Path:
    return Settings.from_env().project_root / "weights" / "rfdetr"


def ensure_rf_home() -> Path:
    """确保 `RF_HOME` 指向一个可写目录，并返回它。"""
    configured = os.environ.get(ENV_KEY, "").strip()
    if configured:
        path = Path(configured).expanduser()
    else:
        path = default_home()
        os.environ[ENV_KEY] = str(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
