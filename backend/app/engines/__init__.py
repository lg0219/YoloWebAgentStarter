"""（lg0219）模型引擎适配层。

目的：把「训练/推理/评估」与具体框架解耦。当前注册 YOLO 与 RF-DETR，
领域服务只依赖 `ModelEngine` 接口；接入新框架时实现接口并注册即可。
"""

from app.engines.base import EngineInfo, ModelEngine, TrainSpec
from app.engines.registry import (
    available_engines,
    configured_engine_key,
    engine_for_weight,
    get_engine,
    register_engine,
)

__all__ = [
    "EngineInfo",
    "ModelEngine",
    "TrainSpec",
    "available_engines",
    "configured_engine_key",
    "engine_for_weight",
    "get_engine",
    "register_engine",
]
