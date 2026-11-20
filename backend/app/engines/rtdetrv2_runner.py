"""独立版 RT-DETRv2 的训练子进程入口。

由 `RtdetrV2Engine.command_prefix()` 通过 `python -m app.engines.rtdetrv2_runner` 调起，
负责：把 YWA 的 YOLO 导出转成 COCO → 生成临时训练配置 → 调用内嵌的 RT-DETRv2 训练 →
把 checkpoints 归位到应用约定的 `weights/best.pt`、`weights/last.pt`。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# variant -> 内嵌配置（相对 vendor 根目录）
VARIANTS = {
    "r18": "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
    "r34": "configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml",
    "r50": "configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",
    "r101": "configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",
}
DEFAULT_VARIANT = "r18"


def vendor_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "rtdetrv2"


def _normalise_variant(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return DEFAULT_VARIANT
    for prefix in ("rtdetrv2-", "rtdetrv2_", "rtdetr-", "rtdetr_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text if text in VARIANTS else DEFAULT_VARIANT


def build_config(
    *,
    variant: str,
    output_dir: Path,
    export_dir: Path,
    data: dict,
    epochs: int,
    batch_size: int,
    resolution: int,
    workers: int,
    backbone: Path | None,
) -> Path:
    root = vendor_root()
    include = (root / VARIANTS[variant]).as_posix()
    train_info = data["splits"]["train"]
    val_info = data["splits"].get("val")
    size = int(resolution)
    train_img = export_dir / "images" / "train"
    val_img = export_dir / "images" / "val"
    lines = [
        "__include__:",
        f"  - '{include}'",
        "",
        f"output_dir: '{output_dir.as_posix()}'",
        f"num_classes: {len(data['names'])}",
        "remap_mscoco_category: False",
        f"eval_spatial_size: [{size}, {size}]",
        f"epoches: {int(epochs)}",
        "",
        "PResNet:",
        # 有预训练主干就用本地路径加载；没有才退化成从零训练
        f"  pretrained: '{backbone.as_posix()}'" if backbone else "  pretrained: False",
        "",
        "train_dataloader:",
        "  dataset:",
        f"    img_folder: '{train_img.as_posix()}'",
        f"    ann_file: '{train_info['path']}'",
        "    transforms:",
        "      ops:",
        "        - {type: RandomHorizontalFlip}",
        f"        - {{type: Resize, size: [{size}, {size}]}}",
        "        - {type: SanitizeBoundingBoxes, min_size: 1}",
        "        - {type: ConvertPILImage, dtype: 'float32', scale: True}",
        "        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}",
        "  collate_fn:",
        "    scales: ~",
        f"  total_batch_size: {int(batch_size)}",
        f"  num_workers: {int(workers)}",
    ]
    if val_info:
        lines += [
            "",
            "val_dataloader:",
            "  dataset:",
            f"    img_folder: '{val_img.as_posix()}'",
            f"    ann_file: '{val_info['path']}'",
            "    transforms:",
            "      ops:",
            f"        - {{type: Resize, size: [{size}, {size}]}}",
            "        - {type: ConvertPILImage, dtype: 'float32', scale: True}",
            f"  num_workers: {int(workers)}",
        ]
    path = output_dir / "ywa_rtdetrv2.yml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _copy_checkpoints(output_dir: Path) -> None:
    weights = output_dir / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    for name in ("best", "last"):
        source = output_dir / f"{name}.pth"
        if source.is_file():
            shutil.copyfile(source, weights / f"{name}.pt")


def _write_sidecar(output_dir: Path, *, variant: str, resolution: int, names: list[str]) -> None:
    """写推理用的旁注文件：记录档位、输入尺寸与类别，供加载模型时还原结构。"""
    weights = output_dir / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    payload = {
        "engine": "rtdetrv2",
        "variant": variant,
        "resolution": int(resolution),
        "num_classes": len(names),
        "names": names,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    for name in ("best", "last"):
        (weights / f"{name}.json").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YWA RT-DETRv2 runner")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--weights", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from app.engines.rtdetrv2_data import convert_export

    data = convert_export(args.dataset_dir, output_dir / "coco")
    print(f"[ywa] coco ready: {data['splits']}", flush=True)

    variant = _normalise_variant(args.variant)

    backbone: Path | None = None
    if args.weights:
        print(f"[ywa] 使用外部权重续训：{args.weights}", flush=True)
    elif not args.no_pretrained:
        from app.engines.rtdetrv2_weights import VARIANT_DEPTH, ensure_backbone

        backbone = ensure_backbone(VARIANT_DEPTH[variant])
    else:
        print("[ywa] 已关闭预训练主干，本次从零训练", flush=True)

    config_path = build_config(
        variant=variant,
        output_dir=output_dir,
        export_dir=Path(args.dataset_dir).resolve(),
        data=data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resolution=args.resolution,
        workers=args.workers,
        backbone=backbone,
    )
    print(f"[ywa] config: {config_path} (variant={variant})", flush=True)

    root = vendor_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # 内嵌版的 training tools 依赖自身包结构，这里直接复用它的入口函数
    from tools.train import main as train_main  # type: ignore[import-not-found]

    train_args = argparse.Namespace(
        config=str(config_path),
        resume=args.weights if args.resume else None,
        tuning=args.weights if (args.weights and not args.resume) else None,
        device=args.device,
        seed=args.seed,
        use_amp=False,
        output_dir=str(output_dir),
        summary_dir=str(output_dir / "tensorboard"),
        test_only=False,
        update=None,
        print_method="builtin",
        print_rank=0,
        local_rank=None,
    )
    train_main(train_args)
    _copy_checkpoints(output_dir)
    _write_sidecar(output_dir, variant=variant, resolution=args.resolution, names=data["names"])
    print("[ywa] checkpoints copied to weights/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
