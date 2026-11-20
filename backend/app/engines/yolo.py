"""YOLO 引擎实现：承载原先散落在 runner / inference 中的 YOLO 细节。

对外行为与改造前一致，仅把实现收敛到适配器里。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from app.core.task_types import TaskType
from app.engines.base import EngineInfo, ModelEngine
from app.models.result_parser import YoloResultParser


class ModelCache:
    """受管 Ultralytics PT 模型的简单 LRU 缓存（最多 3 个）。"""

    _cache: dict[str, Any] = {}
    _last_access: dict[str, float] = {}
    _max_size = 3

    @classmethod
    def get_model(cls, model_id: str, model_path: Path) -> Any:
        from ultralytics import YOLO

        now = time.time()
        if model_id in cls._cache:
            cls._last_access[model_id] = now
            return cls._cache[model_id]
        if len(cls._cache) >= cls._max_size:
            oldest = min(cls._last_access, key=cls._last_access.get)
            cls._cache.pop(oldest, None)
            cls._last_access.pop(oldest, None)
        model = YOLO(str(model_path))
        cls._cache[model_id] = model
        cls._last_access[model_id] = now
        return model


def configured_executable() -> str:
    """读取设置页保存的 YOLO 可执行文件路径（settings.json 的 runtime 段）。"""
    try:
        from app.core.config import Settings

        path = Settings.from_env().data_dir / "settings.json"
        if not path.is_file():
            return ""
        data = json.loads(path.read_text(encoding="utf-8"))
        value = (data.get("runtime") or {}).get("yolo_executable") or ""
        return str(value).strip()
    except Exception:
        # 设置文件损坏或不可读时退回默认解析，不影响训练
        return ""


def resolve_command_prefix() -> list[str]:
    override = os.getenv("YWA_YOLO_EXECUTABLE", "").strip()
    if override:
        return [override]
    configured = configured_executable()
    if configured and Path(configured).expanduser().exists():
        return [str(Path(configured).expanduser())]
    repository_root = Path(__file__).resolve().parents[3]
    binary = "yolo.exe" if os.name == "nt" else "yolo"
    candidate = repository_root / ".venv" / ("Scripts" if os.name == "nt" else "bin") / binary
    if candidate.is_file():
        return [str(candidate)]
    found = shutil.which("yolo")
    if found:
        return [found]
    return [sys.executable, "-c", "from ultralytics.cfg import entrypoint; raise SystemExit(entrypoint())"]


class YoloEngine(ModelEngine):
    info = EngineInfo(
        key="yolo",
        display_name="Ultralytics YOLO",
        task_types=(TaskType.DETECT, TaskType.SEGMENT, TaskType.OBB, TaskType.CLASSIFY),
        weight_suffixes=(".pt", ".onnx", ".yaml"),
    )

    def command_prefix(self) -> list[str]:
        return resolve_command_prefix()

    def build_train_argv(self, spec) -> list[str]:
        # 复用既有命令构造逻辑，保持与改造前完全一致的参数
        from app.training.config import build_training_command

        command = build_training_command(
            task_type=spec.task_type,
            model=spec.model,
            data_yaml=spec.data,
            run_dir=spec.run_dir,
            epochs=spec.epochs,
            img_size=spec.img_size,
            batch_size=spec.batch_size,
            device=spec.device,
            workers=spec.workers,
            seed=spec.seed,
            optimizer=spec.optimizer,
            lr0=spec.lr0,
            patience=spec.patience,
            resume=spec.resume,
            deterministic=spec.deterministic,
        )
        return list(command.args)

    def load_model(self, model_id: str, model_path: Path) -> Any:
        return ModelCache.get_model(model_id, model_path)

    def infer(
        self,
        model: Any,
        image_path: Path,
        confidence: float,
        iou: float,
        task_type: str,
    ) -> list:
        results = model.predict(source=str(image_path), conf=confidence, iou=iou, verbose=False)
        return YoloResultParser().parse(results, task_type)
