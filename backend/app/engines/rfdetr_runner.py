"""RF-DETR 训练子进程入口：由后端以 `python -m app.engines.rfdetr_runner ...` 拉起。

之所以单独成模块：RF-DETR 只提供 Python API，需要把参数翻译成 train() 调用，
并在子进程里跑，保持与 YOLO 相同的「训练在独立进程」模型。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF-DETR training runner")
    parser.add_argument("--dataset-dir", required=True, help="包含 data.yaml 的数据集目录")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=0, help="0 表示使用模型变体默认分辨率")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", default="", help="本地权重路径；为空则使用预训练默认权重")
    parser.add_argument("--variant", default="", help="模型档位：nano / small / base / medium / large")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _build_model(variant: str, weights: str):
    """按档位实例化；档位缺省时退回 Base，本地权重优先。"""
    import rfdetr

    from app.engines.rfdetr_home import ensure_rf_home

    # 框架默认会把权重放进用户目录，受限环境下不可写；统一改到项目内
    ensure_rf_home()

    classes = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "base": "RFDETRBase",
        "medium": "RFDETRMedium",
        "large": "RFDETRLarge",
    }
    key = (variant or "").strip().lower()
    class_name = classes.get(key, "RFDETRBase")
    model_cls = getattr(rfdetr, class_name)
    path = (weights or "").strip()
    if path and Path(path).expanduser().is_file():
        return model_cls(pretrain_weights=str(Path(path).expanduser()))
    # 不传 pretrain_weights：由框架按档位下载官方预训练权重
    return model_cls()


def _runtime_kwargs(raw: str) -> dict[str, object]:
    """把应用的设备选择翻译成 Lightning 认的 accelerator / devices。"""
    value = (raw or "").strip().lower()
    if value in ("", "auto"):
        return {"accelerator": "gpu", "devices": 1} if _cuda_available() else {"accelerator": "cpu", "devices": 1}
    if value == "cpu":
        return {"accelerator": "cpu", "devices": 1}
    if value == "mps":
        return {"accelerator": "mps", "devices": 1}
    ids = [part for part in value.replace("cuda:", "").split(",") if part.strip().isdigit()]
    if ids:
        devices: object = [int(item) for item in ids]
        if len(devices) == 1:
            devices = 1
        return {"accelerator": "gpu", "devices": devices}
    return {"accelerator": "cpu", "devices": 1}


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _class_names(dataset_dir: Path) -> list[str] | None:
    """从 data.yaml 读取类别名；RF-DETR 只认 class_names，不认 num_classes。"""
    data_file = dataset_dir / "data.yaml"
    if not data_file.is_file():
        return None
    try:
        import yaml

        content = yaml.safe_load(data_file.read_text(encoding="utf-8")) or {}
        names = content.get("names")
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
        if isinstance(names, list):
            return [str(item) for item in names]
    except Exception:
        return None
    return None


def _normalise_outputs(output_dir: Path, variant: str = "") -> None:
    """把 RF-DETR 的产物归位到应用约定：weights/{best,last}.pt 与 results.csv。"""
    import csv
    import shutil

    weights = output_dir / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    for name, candidates in (
        ("best.pt", ("checkpoint_best_ema.pth", "checkpoint_best_total.pth", "last_ema.pth")),
        ("last.pt", ("last_ema.pth", "last.ckpt")),
    ):
        for candidate in candidates:
            source = output_dir / candidate
            if source.is_file():
                shutil.copyfile(source, weights / name)
                break
    # 旁注：告诉应用这个 .pt 是 RF-DETR 产物，推理时按 RF-DETR 加载
    marker = json.dumps(
        {"engine": "rfdetr", "variant": (variant or "base").strip().lower() or "base"},
        ensure_ascii=False,
        indent=2,
    )
    for name in ("best", "last"):
        (weights / f"{name}.json").write_text(marker, encoding="utf-8")

    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.is_file():
        return
    rows: dict[int, dict[str, str]] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                epoch = int(float(row.get("epoch") or 0))
            except (TypeError, ValueError):
                continue
            bucket = rows.setdefault(epoch, {})
            for key, value in row.items():
                if key in ("epoch", "step") or value in ("", None):
                    continue
                bucket[key] = value

    def pick(bucket: dict[str, str], *keys: str) -> str:
        for key in keys:
            value = bucket.get(key)
            if value not in (None, ""):
                return value
        return ""

    target = output_dir / "results.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "time",
                "train/loss",
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            ]
        )
        for epoch in sorted(rows):
            bucket = rows[epoch]
            writer.writerow(
                [
                    epoch + 1,
                    "",
                    pick(bucket, "train/loss"),
                    pick(bucket, "val/precision"),
                    pick(bucket, "val/recall"),
                    pick(bucket, "val/ema_mAP_50", "val/mAP_50"),
                    pick(bucket, "val/ema_mAP_50_95", "val/mAP_50_95"),
                ]
            )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import rfdetr  # noqa: F401
    except ImportError:
        print(
            "[rfdetr_runner] 未安装 rfdetr 依赖，请执行：pip install -r backend/requirements-detr.txt",
            file=sys.stderr,
        )
        return 2
    model = _build_model(args.variant, args.weights)
    print(f"[rfdetr_runner] variant={(args.variant or 'base')}", flush=True)

    train_kwargs: dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "dataset_file": "yolo",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        # vernier 后端依赖 pyarrow，而 pyarrow 新版要求 numpy>=2；本项目为兼容 ultralytics 锁在 1.x，
        # 因此固定用 faster-coco-eval 后端（指标一致）。
        "eval_backend": "faster_coco_eval",
        **_runtime_kwargs(args.device),
    }
    if args.resume:
        train_kwargs["resume"] = str(output_dir / "checkpoint.pth")
    names = _class_names(dataset_dir)
    if names:
        train_kwargs["class_names"] = names

    print(f"[rfdetr_runner] dataset_dir={dataset_dir}", flush=True)
    print(f"[rfdetr_runner] output_dir={output_dir}", flush=True)
    print(f"[rfdetr_runner] train_kwargs={train_kwargs}", flush=True)
    model.train(**train_kwargs)
    _normalise_outputs(output_dir, args.variant)
    print("[rfdetr_runner] training finished", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
