"""把 YWA 的 YOLO 导出目录转换成独立版 RT-DETR 需要的 COCO 标注。

独立版 RT-DETR 与 Ultralytics 有三处口径不同，这里一次处理掉：

1. 类别编号必须从 0 开始（它不做类别重映射）；
2. 训练 / 验证要各自一个 json（它按 split 分别读取）；
3. json 写成纯 ASCII（Windows 上它按系统编码读标注，中文会直接报错）。
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def _class_names(export_dir: Path) -> list[str]:
    path = export_dir / "data.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names") or []
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    return [str(item) for item in names]


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as handle:
        return handle.size


def _annotations_for(label_path: Path, width: int, height: int) -> list[list[float]]:
    """读取 YOLO 标签，返回 [class_index, x, y, w, h]（绝对像素）。"""
    if not label_path.is_file():
        return []
    rows: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_index = int(float(parts[0]))
            cx, cy, box_w, box_h = (float(value) for value in parts[1:5])
        except ValueError:
            continue
        rows.append(
            [
                class_index,
                round(cx * width - box_w * width / 2, 2),
                round(cy * height - box_h * height / 2, 2),
                round(box_w * width, 2),
                round(box_h * height, 2),
            ]
        )
    return rows


def _build_split(export_dir: Path, split: str, names: list[str]) -> dict | None:
    image_dir = export_dir / "images" / split
    label_dir = export_dir / "labels" / split
    if not image_dir.is_dir():
        return None

    images: list[dict] = []
    annotations: list[dict] = []
    for index, image_path in enumerate(sorted(image_dir.iterdir()), start=1):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        width, height = _image_size(image_path)
        images.append({"id": index, "file_name": image_path.name, "width": width, "height": height})
        for class_index, x, y, box_w, box_h in _annotations_for(label_dir / f"{image_path.stem}.txt", width, height):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": index,
                    # 独立版 RT-DETR 直接用 category_id 当标签，必须 0 基
                    "category_id": class_index,
                    "bbox": [x, y, box_w, box_h],
                    "area": round(box_w * box_h, 2),
                    "iscrowd": 0,
                }
            )
    if not images:
        return None
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index, "name": name} for index, name in enumerate(names)],
    }


def convert_export(export_dir: str | Path, out_dir: str | Path) -> dict:
    """转换导出的 YOLO 数据集，返回各 split 的 json 路径与统计。"""
    source = Path(export_dir).resolve()
    target = Path(out_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    names = _class_names(source)

    result: dict = {"names": names, "splits": {}}
    for split in ("train", "val"):
        payload = _build_split(source, split, names)
        if payload is None:
            continue
        path = target / f"{split}.json"
        # ensure_ascii：中文类名写成转义形式，避开它按 GBK 读 json 的问题
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="ascii")
        result["splits"][split] = {
            "path": str(path),
            "images": len(payload["images"]),
            "annotations": len(payload["annotations"]),
        }
    if "train" not in result["splits"]:
        raise ValueError(f"no train images found under {source / 'images' / 'train'}")
    return result
