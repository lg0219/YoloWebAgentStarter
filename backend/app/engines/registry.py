"""引擎注册表：默认 YOLO；导入时注册 RF-DETR，可按名称或权重文件后缀取用。"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.errors import ValidationError
from app.engines.base import EngineInfo, ModelEngine

DEFAULT_ENGINE_KEY = "yolo"

_ENGINES: dict[str, ModelEngine] = {}


def register_engine(engine: ModelEngine) -> ModelEngine:
    _ENGINES[engine.info.key] = engine
    return engine


def _bootstrap() -> None:
    if _ENGINES:
        return
    # 延迟导入，避免与领域模块形成循环依赖
    from app.engines.rfdetr import RfDetrEngine
    from app.engines.rtdetrv2 import RtdetrV2Engine
    from app.engines.yolo import YoloEngine

    register_engine(YoloEngine())
    register_engine(RfDetrEngine())
    register_engine(RtdetrV2Engine())


def get_engine(key: str | None = None) -> ModelEngine:
    _bootstrap()
    resolved = (key or DEFAULT_ENGINE_KEY).strip() or DEFAULT_ENGINE_KEY
    engine = _ENGINES.get(resolved)
    if engine is None:
        raise ValidationError(
            "engine_not_registered",
            f"模型引擎未注册：{resolved}",
            details={"available": sorted(_ENGINES)},
        )
    return engine


def engine_for_weight(path: str | Path, fallback: str | None = None) -> ModelEngine:
    """按权重文件后缀推断引擎；无法判断时退回指定引擎或默认引擎。"""
    suffix = Path(str(path)).suffix.lower()
    _bootstrap()
    # .pt / .pth 都可能是独立版 RT-DETR 的产物；带训练旁注文件的优先判给它
    if suffix in {".pt", ".pth"}:
        from app.engines.rtdetrv2 import read_sidecar

        if read_sidecar(Path(str(path))) is not None:
            return get_engine("rtdetrv2")
    for engine in _ENGINES.values():
        if suffix in engine.info.weight_suffixes:
            return engine
    return get_engine(fallback)


def available_engines() -> list[EngineInfo]:
    _bootstrap()
    return [engine.info for engine in _ENGINES.values()]


def engine_capabilities() -> list[dict]:
    """返回引擎能力描述，供前端按能力渲染训练参数。"""
    _bootstrap()
    payload: list[dict] = []
    for engine in _ENGINES.values():
        payload.append(
            {
                "key": engine.info.key,
                "display_name": engine.info.display_name,
                "task_types": [task.value for task in engine.info.task_types],
                "weight_suffixes": list(engine.info.weight_suffixes),
                "supported_params": sorted(engine.supported_params),
                "fixed_params": dict(engine.fixed_params),
                "allows_blank_model": bool(getattr(engine, "allows_blank_model", False)),
                "model_choices": list(getattr(engine, "model_choices", ())),
            }
        )
    return payload


def configured_engine_key(data_dir: Path | None) -> str:
    """训练引擎由训练请求指定；未指定时使用默认引擎。

    历史版本的设置页曾把引擎写入 settings.json 的 runtime.model_engine，
    这里仍兼容读取，但新版本不再写该字段。
    """
    try:
        if data_dir is not None:
            path = Path(data_dir) / "settings.json"
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                value = str((data.get("runtime") or {}).get("model_engine") or "").strip()
                if value:
                    return value
    except Exception:
        pass
    return DEFAULT_ENGINE_KEY
