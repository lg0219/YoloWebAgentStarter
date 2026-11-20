"""RF-DETR 训练子进程入口：由后端以 `python -m app.engines.rfdetr_runner ...` 拉起。

之所以单独成模块：RF-DETR 只提供 Python API，需要把参数翻译成 train() 调用，
并在子进程里跑，保持与 YOLO 相同的「训练在独立进程」模型。
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _normalize_device(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value in ("", "auto"):
        return "cuda" if _cuda_available() else "cpu"
    if value.isdigit():
        return f"cuda:{value}" if _cuda_available() else "cpu"
    return value


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _count_classes(dataset_dir: Path) -> int | None:
    """从 data.yaml 读取类别数，读不到就交给 RF-DETR 自行对齐。"""
    data_file = dataset_dir / "data.yaml"
    if not data_file.is_file():
        return None
    try:
        import yaml

        content = yaml.safe_load(data_file.read_text(encoding="utf-8")) or {}
        names = content.get("names")
        if isinstance(names, dict):
            return len(names)
        if isinstance(names, list):
            return len(names)
    except Exception:
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from rfdetr import RFDETRBase
    except ImportError:
        print(
            "[rfdetr_runner] 未安装 rfdetr 依赖，请执行：pip install -r backend/requirements-detr.txt",
            file=sys.stderr,
        )
        return 2
    weights = (args.weights or "").strip()
    if weights and Path(weights).expanduser().is_file():
        model = RFDETRBase(pretrain_weights=str(Path(weights).expanduser()))
    else:
        model = RFDETRBase()

    train_kwargs: dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "dataset_file": "yolo",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "device": _normalize_device(args.device),
        "seed": int(args.seed),
    }
    if args.resume:
        train_kwargs["resume"] = str(output_dir / "checkpoint.pth")
    num_classes = _count_classes(dataset_dir)
    if num_classes:
        train_kwargs["num_classes"] = num_classes

    print(f"[rfdetr_runner] dataset_dir={dataset_dir}", flush=True)
    print(f"[rfdetr_runner] output_dir={output_dir}", flush=True)
    print(f"[rfdetr_runner] train_kwargs={train_kwargs}", flush=True)
    model.train(**train_kwargs)
    print("[rfdetr_runner] training finished", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
