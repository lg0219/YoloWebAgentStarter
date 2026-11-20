"""RF-DETR 引擎实现（接入示例）。

说明：RF-DETR 没有 CLI，训练通过 `app.engines.rfdetr_runner` 子进程完成；
推理直接调用其 Python API。当前仅支持 detect 任务。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from app.core.errors import ValidationError
from app.core.task_types import TaskType
from app.engines.base import EngineInfo, ModelEngine, TrainSpec
from app.models.result_parser import Detection


def _resolve_source_root() -> Path:
    """定位仓库根目录，用于给子进程设置 PYTHONPATH。"""
    return Path(__file__).resolve().parents[3]


class RfDetrEngine(ModelEngine):
    info = EngineInfo(
        key="rfdetr",
        display_name="RF-DETR",
        task_types=(TaskType.DETECT,),
        weight_suffixes=(".pth",),
    )

    # RF-DETR 只接受这些训练参数；分辨率与优化器由框架自身决定
    supported_params = frozenset({"model", "epochs", "batch_size", "device", "seed"})
    fixed_params = {"img_size": "使用模型变体默认分辨率", "optimizer": "框架内部默认", "workers": "不支持多进程加载"}

    def command_prefix(self) -> list[str]:
        return [sys.executable, "-m", "app.engines.rfdetr_runner"]

    def build_train_argv(self, spec: TrainSpec) -> list[str]:
        if spec.task_type is not TaskType.DETECT:
            raise ValidationError(
                "engine_task_unsupported",
                "RF-DETR 目前只支持 detect 任务。",
                details={"task_type": spec.task_type.value},
            )
        argv = [
            self.info.key,
            "--dataset-dir", str(Path(spec.data).resolve().parent),
            "--output-dir", spec.run_dir,
            "--epochs", str(spec.epochs),
            "--batch-size", str(spec.batch_size),
            "--resolution", str(spec.img_size),
            "--device", spec.device,
            "--seed", str(spec.seed),
        ]
        if spec.model:
            argv.extend(["--weights", spec.model])
        if spec.resume:
            argv.append("--resume")
        return argv

    def load_model(self, model_id: str, model_path: Path) -> Any:
        try:
            from rfdetr import RFDETRBase
        except ImportError as exc:  # 依赖是可选安装
            raise ValidationError(
                "engine_dependency_missing",
                "未安装 RF-DETR 依赖，请执行：pip install -r backend/requirements-detr.txt",
            ) from exc
        return RFDETRBase(pretrain_weights=str(model_path))

    def infer(
        self,
        model: Any,
        image_path: Path,
        confidence: float,
        iou: float,
        task_type: str,
    ) -> list:
        if task_type != TaskType.DETECT.value:
            raise ValidationError("engine_task_unsupported", "RF-DETR 目前只支持 detect 推理。")
        # RF-DETR 的 threshold 即置信度阈值；iou 参数其 API 未暴露，暂不参与
        result = model.predict(str(image_path), threshold=float(confidence))
        return _to_detections(result)


def _to_detections(result: Any) -> list[Detection]:
    """把 RF-DETR（supervision Detections）结果映射为 YWA 的统一结构。"""
    xyxy = getattr(result, "xyxy", None)
    confidences = getattr(result, "confidence", None)
    class_ids = getattr(result, "class_id", None)
    if xyxy is None:
        return []
    detections: list[Detection] = []
    for index, box in enumerate(xyxy):
        x1, y1, x2, y2 = (float(value) for value in box[:4])
        width, height = x2 - x1, y2 - y1
        if width <= 1e-4 or height <= 1e-4:
            continue
        score = float(confidences[index]) if confidences is not None else 1.0
        class_index = int(class_ids[index]) if class_ids is not None else 0
        detections.append(
            Detection(
                class_index=class_index,
                confidence=score,
                x=x1,
                y=y1,
                width=width,
                height=height,
            )
        )
    return detections
