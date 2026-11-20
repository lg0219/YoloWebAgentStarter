"""独立版 RT-DETRv2 引擎（内嵌源码，仅支持检测）。

与 Ultralytics 版 RT-DETR 的区别：代码来自 lyuwenyu/RT-DETR 的 `rtdetrv2_pytorch`，
数据要转成 COCO，权重是 `.pth`，没有 pip 包，也没有 CLI——训练走 `rtdetrv2_runner` 子进程。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from app.core.errors import ValidationError
from app.core.task_types import TaskType
from app.engines.base import EngineInfo, ModelEngine, TrainSpec

# 可选的结构档位，作为「基础权重」字段的取值
VARIANT_CHOICES = ("r18", "r34", "r50", "r101")
DEFAULT_VARIANT = "r18"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class RtdetrV2Engine(ModelEngine):
    info = EngineInfo(
        key="rtdetrv2",
        display_name="RT-DETR（独立版 v2）",
        task_types=(TaskType.DETECT,),
        weight_suffixes=(".pth",),
    )

    # 独立版按自己的配置跑，界面上的这些参数实际生效
    supported_params = frozenset({"model", "epochs", "batch_size", "img_size", "device", "workers", "seed"})
    fixed_params = {
        "optimizer": "由内置配置决定（AdamW）",
        "lr0": "由内置配置决定",
        "patience": "不支持早停",
        "deterministic": "内置配置未暴露",
    }

    # 允许「基础权重」留空（或用档位名），表示按该档位从零训练
    allows_blank_model = True
    model_choices = VARIANT_CHOICES

    def command_prefix(self) -> list[str]:
        return [sys.executable, "-m", "app.engines.rtdetrv2_runner"]

    def build_train_argv(self, spec: TrainSpec) -> list[str]:
        if spec.task_type is not TaskType.DETECT:
            raise ValidationError(
                "engine_task_unsupported",
                "RT-DETR（独立版 v2）只支持 detect 任务。",
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
            "--workers", str(max(spec.workers, 0)),
        ]
        model = (spec.model or "").strip()
        if model:
            resolved = Path(model).expanduser()
            if resolved.is_file():
                argv.extend(["--weights", str(resolved.resolve())])
            else:
                argv.extend(["--variant", model])
        if spec.resume:
            argv.append("--resume")
        return argv

    def load_model(self, model_id: str, model_path: Path) -> Any:
        raise ValidationError(
            "engine_inference_unsupported",
            "RT-DETR（独立版 v2）当前只支持训练，推理请使用 Ultralytics 版 RT-DETR。",
        )

    def infer(
        self,
        model: Any,
        image_path: Path,
        confidence: float,
        iou: float,
        task_type: str,
    ) -> list:
        raise ValidationError(
            "engine_inference_unsupported",
            "RT-DETR（独立版 v2）当前只支持训练，推理请使用 Ultralytics 版 RT-DETR。",
        )
