from __future__ import annotations

import random
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.core.ids import new_id
from app.core.models import ImageItem
from app.core.schemas import AutoSplitRequest, BulkImageSplitUpdate, ScanImagesResponse, SplitName, SplitOperationResponse
from app.core.storage import SUPPORTED_IMAGE_SUFFIXES, Storage
from app.dataset.dcm import convert_dicom_to_png, find_sidecar_json, is_dicom_name
from app.dataset.service import get_dataset, refresh_dataset_counts


def get_image(session: Session, image_id: str) -> ImageItem:
    image = session.get(ImageItem, image_id)
    if image is None:
        raise NotFoundError("image_not_found", "Image was not found.")
    return image


def list_images(session: Session, dataset_id: str) -> list[ImageItem]:
    get_dataset(session, dataset_id)
    query = select(ImageItem).where(ImageItem.dataset_id == dataset_id).order_by(ImageItem.created_at, ImageItem.id)
    return list(session.scalars(query))


def _stage_image(
    session: Session,
    storage: Storage,
    dataset_id: str,
    original_name: str,
    content: bytes,
    split: SplitName,
    sidecar: bytes | None = None,
) -> ImageItem:
    # （lg0219）DICOM 在写入存储前先转成 8bit PNG，后续链路只认识普通图片。
    if is_dicom_name(original_name):
        converted = convert_dicom_to_png(content, sidecar)
        content = converted.png
        original_name = f"{Path(original_name).stem}.png"
    image_id = new_id("img")
    storage_name = storage.safe_storage_name(original_name, image_id)
    width, height = storage.write_image(dataset_id, storage_name, content)
    image = ImageItem(
        id=image_id,
        dataset_id=dataset_id,
        file_name=Path(original_name).name,
        storage_name=storage_name,
        width=width,
        height=height,
        split=split,
        status="unannotated",
    )
    session.add(image)
    return image


def add_uploaded_images(
    session: Session,
    storage: Storage,
    dataset_id: str,
    uploads: list[tuple[str, bytes]],
    split: SplitName,
    sidecars: dict[str, bytes] | None = None,
) -> list[ImageItem]:
    get_dataset(session, dataset_id)
    staged: list[ImageItem] = []
    try:
        for file_name, content in uploads:
            sidecar = (sidecars or {}).get(file_name)
            staged.append(_stage_image(session, storage, dataset_id, file_name, content, split, sidecar))
        session.flush()
        refresh_dataset_counts(session, dataset_id)
        session.commit()
    except Exception:
        session.rollback()
        for image in staged:
            storage.image_path(dataset_id, image.storage_name).unlink(missing_ok=True)
        raise
    for image in staged:
        session.refresh(image)
    return staged


def scan_images(
    session: Session,
    storage: Storage,
    dataset_id: str,
    requested_path: str,
    recursive: bool,
    split: SplitName,
) -> ScanImagesResponse:
    get_dataset(session, dataset_id)
    root = storage.resolve_import_directory(requested_path)
    paths = sorted(path for path in (root.rglob("*") if recursive else root.iterdir()) if path.is_file())
    total_found = len(paths)
    skipped = 0
    invalid = 0
    uploads: list[tuple[str, bytes]] = []
    sidecars: dict[str, bytes] = {}
    for path in paths:
        # （lg0219）.dcm / .dicom 也要纳入扫描，导入时统一转 PNG。
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES and not is_dicom_name(path.name):
            skipped += 1
            continue
        try:
            resolved = storage.resolve_import_file(path)
            uploads.append((path.name, resolved.read_bytes()))
            if is_dicom_name(path.name):
                sidecar_path = find_sidecar_json(resolved)
                if sidecar_path is not None:
                    sidecars[path.name] = sidecar_path.read_bytes()
        except (OSError, ValidationError):
            invalid += 1
    imported = 0
    for file_name, content in uploads:
        try:
            add_uploaded_images(session, storage, dataset_id, [(file_name, content)], split, sidecars)
            imported += 1
        except ValidationError:
            invalid += 1
    return ScanImagesResponse(total_found=total_found, imported=imported, skipped=skipped, invalid=invalid)


def update_image_split(session: Session, image_id: str, split: SplitName) -> ImageItem:
    image = get_image(session, image_id)
    image.split = split
    session.commit()
    session.refresh(image)
    return image


def split_counts(session: Session, dataset_id: str) -> dict[SplitName, int]:
    counts: dict[SplitName, int] = {"train": 0, "val": 0, "test": 0}
    counts.update(dict(session.execute(select(ImageItem.split, func.count(ImageItem.id)).where(ImageItem.dataset_id == dataset_id).group_by(ImageItem.split)).all()))
    return counts


def update_image_splits(session: Session, dataset_id: str, payload: BulkImageSplitUpdate) -> SplitOperationResponse:
    get_dataset(session, dataset_id)
    ids = list(dict.fromkeys(payload.image_ids))
    images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == dataset_id, ImageItem.id.in_(ids))))
    if len(images) != len(ids):
        raise NotFoundError("image_not_found", "One or more images were not found in this dataset.")
    for image in images:
        image.split = payload.split
    session.commit()
    return SplitOperationResponse(updated=len(images), split_counts=split_counts(session, dataset_id))


def auto_split_images(session: Session, dataset_id: str, payload: AutoSplitRequest) -> SplitOperationResponse:
    """Adapt upstream DatasetSplitProcessor's deterministic target-balancing for Starter's flat image model."""
    get_dataset(session, dataset_id)
    images = list(session.scalars(select(ImageItem).where(ImageItem.dataset_id == dataset_id).order_by(ImageItem.id)))
    random.Random(payload.seed).shuffle(images)
    ratios = {"train": payload.train_ratio, "val": payload.val_ratio, "test": payload.test_ratio}
    targets = {split: ratio * len(images) for split, ratio in ratios.items()}
    counts = {split: 0 for split in ratios}
    for image in images:
        candidates = [split for split, ratio in ratios.items() if ratio > 0] or ["train"]
        split = min(candidates, key=lambda name: (counts[name] / max(targets[name], 1), ("train", "val", "test").index(name)))
        image.split = split
        counts[split] += 1
    session.commit()
    return SplitOperationResponse(updated=len(images), split_counts=split_counts(session, dataset_id))


def delete_image(session: Session, storage: Storage, image_id: str) -> None:
    image = get_image(session, image_id)
    dataset_id = image.dataset_id
    path = storage.image_path(dataset_id, image.storage_name)
    session.delete(image)
    session.flush()
    refresh_dataset_counts(session, dataset_id)
    session.commit()
    path.unlink(missing_ok=True)
