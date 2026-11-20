"""独立版 RT-DETRv2 引擎（内嵌源码，仅支持检测）。

与 Ultralytics 版 RT-DETR 的区别：代码来自 lyuwenyu/RT-DETR 的 `rtdetrv2_pytorch`，
数据要转成 COCO，权重是 `.pth`，没有 pip 包，也没有 CLI——训练走 `rtdetrv2_runner` 子进程。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from app.core.errors import ValidationError
from app.core.task_types import TaskType
from app.engines.base import EngineInfo, ModelEngine, TrainSpec
from app.models.result_parser import Detection

# 可选的结构档位，作为「基础权重」字段的取值
VARIANT_CHOICES = ("r18", "r34", "r50", "r101")
DEFAULT_VARIANT = "r18"

# 推理用的 ImageNet 归一化（与上游 val 管线一致）
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _vendor_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "rtdetrv2"


def _ensure_vendor_on_path() -> Path:
    root = _vendor_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# 档位 -> 内嵌配置（相对 vendor 根目录），与 rtdetrv2_runner.VARIANTS 保持一致
VARIANT_CONFIGS = {
    "r18": "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
    "r34": "configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml",
    "r50": "configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",
    "r101": "configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",
}


def read_sidecar(model_path: Path) -> dict | None:
    """读取训练时写下的旁注文件（档位 / 输入尺寸 / 类别）。"""
    for candidate in (model_path.with_suffix(".json"), model_path.parent / "best.json"):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if data.get("engine") == "rtdetrv2":
                return data
    return None


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
        """从训练产物重建模型：旁注文件决定档位、输入尺寸与类别数。"""
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise ValidationError("engine_weight_missing", f"权重文件不存在：{path}")
        sidecar = read_sidecar(path)
        if sidecar is None:
            raise ValidationError(
                "engine_sidecar_missing",
                "缺少训练时生成的 best.json，无法还原模型结构。请用本引擎重新训练。",
            )

        _ensure_vendor_on_path()
        try:
            import torch
            from src.core import YAMLConfig
        except ImportError as exc:
            raise ValidationError(
                "engine_dependency_missing",
                "未安装 RT-DETR（独立版）依赖，请执行：pip install -r backend/requirements-rtdetrv2.txt",
            ) from exc

        variant = sidecar.get("variant") or DEFAULT_VARIANT
        resolution = int(sidecar.get("resolution") or 640)
        num_classes = int(sidecar.get("num_classes") or 1)
        config_path = _vendor_root() / VARIANT_CONFIGS.get(variant, VARIANT_CONFIGS[DEFAULT_VARIANT])

        cfg = YAMLConfig(
            str(config_path),
            num_classes=num_classes,
            remap_mscoco_category=False,
            eval_spatial_size=[resolution, resolution],
            PResNet={"pretrained": False},
        )
        model = cfg.model.eval()
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        state = checkpoint.get("ema") or checkpoint.get("model") or checkpoint
        model.load_state_dict(state, strict=False)
        return {"model": model, "postprocessor": cfg.postprocessor, "resolution": resolution, "names": sidecar.get("names") or []}

    def infer(
        self,
        model: Any,
        image_path: Path,
        confidence: float,
        iou: float,
        task_type: str,
    ) -> list:
        if task_type != TaskType.DETECT.value:
            raise ValidationError("engine_task_unsupported", "RT-DETR（独立版 v2）只支持检测推理。")

        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        bundle = model
        net = bundle["model"]
        resolution = bundle["resolution"]
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            tensor = TF.pil_to_tensor(rgb).float() / 255.0
        tensor = TF.resize(tensor, [resolution, resolution], antialias=True)
        tensor = TF.normalize(tensor, mean=_MEAN, std=_STD).unsqueeze(0)

        with torch.no_grad():
            outputs = net(tensor)
            sizes = torch.tensor([[width, height]])
            results = bundle["postprocessor"](outputs, sizes)

        # 上游后处理按批返回 list[dict]，这里固定单张图
        resolved = results[0] if isinstance(results, list) else results
        labels = resolved["labels"].tolist()
        boxes = resolved["boxes"].tolist()
        scores = resolved["scores"].tolist()

        detections: list[Detection] = []
        for label, box, score in zip(labels, boxes, scores, strict=True):
            if score < confidence:
                continue
            x1, y1, x2, y2 = box
            box_w, box_h = x2 - x1, y2 - y1
            if box_w <= 1e-4 or box_h <= 1e-4:
                continue
            detections.append(
                Detection(
                    class_index=int(label),
                    confidence=float(score),
                    x=float(x1),
                    y=float(y1),
                    width=float(box_w),
                    height=float(box_h),
                )
            )
        return detections
