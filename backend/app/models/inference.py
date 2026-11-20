from __future__ import annotations

import tempfile
import time
from pathlib import Path

from app.core.errors import ValidationError
from app.engines.registry import engine_for_weight
from app.engines.yolo import ModelCache  # 兼容既有调用方与测试（YOLO 模型缓存实现）

__all__ = ["ModelCache", "run_managed_inference", "run_test_inference"]


def run_test_inference(*, model_id: str, model_path: Path, task_type: str, image_bytes: bytes, confidence: float, iou: float, class_names: dict[int, str]) -> dict:
    """Adapted from upstream ModelVersionService.run_test_inference without external model paths or save-test."""
    suffix = ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(image_bytes)
        temporary_path = Path(temporary.name)
    try:
        started = time.perf_counter()
        actual_confidence = min(confidence, 0.01) if confidence > 0 else 0.01
        result = _serialize_inference_result(model_id, task_type, class_names, _run_inference(model_id, model_path, task_type, temporary_path, actual_confidence, iou))
        result["inference_time_ms"] = (time.perf_counter() - started) * 1000
        return result
    except Exception as exc:
        raise ValidationError("inference_failed", f"Local model inference failed: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def run_managed_inference(*, model_id: str, model_path: Path, task_type: str, image_path: Path, confidence: float, iou: float, class_names: dict[int, str]) -> list:
    """Run the same cached engine path used by model quick-test on a managed image."""

    return _run_inference(model_id, model_path, task_type, image_path, confidence, iou)


def _run_inference(model_id: str, model_path: Path, task_type: str, image_path: Path, confidence: float, iou: float) -> list:
    """按权重文件后缀选择引擎推理：.pt 走 YOLO，.pth 走 RF-DETR。"""
    engine = engine_for_weight(model_path)
    model = engine.load_model(model_id, model_path)
    return engine.infer(model, image_path, confidence, iou, task_type)


def _serialize_inference_result(model_id: str, task_type: str, class_names: dict[int, str], detections: list) -> dict:
    return {
        "model_id": model_id,
        "task_type": task_type,
        "detections": [
            {
                "class_index": item.class_index,
                "class_name": class_names.get(item.class_index, f"class_{item.class_index}"),
                "confidence": item.confidence,
                "x": item.x,
                "y": item.y,
                "width": item.width,
                "height": item.height,
                "polygon": [list(point) for point in item.polygon] if item.polygon else None,
                "obb_points": [list(point) for point in item.obb_points] if item.obb_points else None,
            }
            for item in detections
        ],
    }
