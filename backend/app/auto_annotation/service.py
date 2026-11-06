from __future__ import annotations

import math

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.errors import NotFoundError, ValidationError
from app.core.ids import new_id
from app.core.models import Annotation, AutoAnnotationTask, ClassLabel, ImageItem, ModelVersion
from app.core.storage import Storage
from app.core.task_types import annotation_type_for
from app.core.time import utc_now
from app.dataset.annotation.geometry import validate_bbox, validate_obb, validate_polygon
from app.dataset.service import get_dataset
from app.models.result_parser import Detection
from app.training.observability.log_store import TrainingLogStore
from app.auto_annotation.schemas import AutoAnnotationCreateRequest, AutoAnnotationLogResponse, AutoAnnotationTaskResponse


def task_response(task: AutoAnnotationTask) -> AutoAnnotationTaskResponse:
    return AutoAnnotationTaskResponse.model_validate(task)


class AutoAnnotationService:
    def __init__(self, session_factory: sessionmaker[Session], storage: Storage, queue) -> None:
        self.session_factory = session_factory
        self.storage = storage
        self.queue = queue

    def create_task(
        self,
        session: Session,
        dataset_id: str,
        payload: AutoAnnotationCreateRequest,
        *,
        task_id: str | None = None,
    ) -> AutoAnnotationTask:
        dataset = get_dataset(session, dataset_id)
        model = session.get(ModelVersion, payload.model_id)
        if model is None:
            raise NotFoundError("model_not_found", "未找到该模型版本。")
        if model.status != "active":
            raise ValidationError("model_archived", "自动标注需要使用未归档的受管模型。")
        if model.format != "pt" or model.engine_type != "ultralytics":
            raise ValidationError("unsupported_model_engine", "自动标注只支持受管的 Ultralytics PT 模型。")
        model_path = self.storage.managed_model_path(model.model_path)
        if not model_path.is_file():
            raise NotFoundError("model_file_missing", "受管模型文件已丢失。")
        if model.task_type != dataset.task_type:
            raise ValidationError("auto_annotation_task_mismatch", "模型的任务类型必须与数据集的任务类型一致。")
        image_query = select(ImageItem).where(ImageItem.dataset_id == dataset_id)
        if payload.skip_annotated_images:
            image_query = image_query.where(~ImageItem.annotations.any())
        images = list(session.scalars(image_query.order_by(ImageItem.created_at, ImageItem.id)))
        if not images:
            message = "数据集中没有未标注的图片。若要处理已有标注的图片，请关闭「跳过已有标注的图片」。" if payload.skip_annotated_images else "数据集中没有可标注的图片。"
            raise ValidationError("auto_annotation_dataset_empty", message)
        target_classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == dataset_id).order_by(ClassLabel.class_index)))
        if not target_classes:
            raise ValidationError("auto_annotation_classes_missing", "请先为数据集添加至少一个类别，再运行自动标注。")
        source_classes = list(
            session.scalars(
                select(ClassLabel).where(ClassLabel.dataset_id == model.dataset_id).order_by(ClassLabel.class_index)
            )
        ) if model.dataset_id else []
        mapping = self._resolve_mapping(payload.class_mapping, target_classes, source_classes)
        if task_id:
            existing = session.get(AutoAnnotationTask, task_id)
            if existing is not None:
                return existing
        resolved_task_id = task_id or new_id("auto")
        log_path = self.storage.auto_annotation_task_dir(resolved_task_id) / "auto_annotation.log"
        task = AutoAnnotationTask(
            id=resolved_task_id,
            dataset_id=dataset.id,
            model_id=model.id,
            task_type=dataset.task_type,
            status="pending",
            clean_old_annotations=payload.clean_old_annotations,
            skip_annotated_images=payload.skip_annotated_images,
            confidence=payload.confidence,
            iou=payload.iou,
            class_mapping=mapping,
            total_images=len(images),
            logs_path=str(log_path),
        )
        session.add(task)
        session.commit()
        session.refresh(task)
        self.queue.submit(task.id)
        return task

    def list_tasks(self, session: Session, dataset_id: str) -> list[AutoAnnotationTask]:
        get_dataset(session, dataset_id)
        return list(
            session.scalars(
                select(AutoAnnotationTask)
                .where(AutoAnnotationTask.dataset_id == dataset_id)
                .order_by(AutoAnnotationTask.created_at.desc(), AutoAnnotationTask.id.desc())
            )
        )

    def get_task(self, session: Session, task_id: str) -> AutoAnnotationTask:
        task = session.get(AutoAnnotationTask, task_id)
        if task is None:
            raise NotFoundError("auto_annotation_task_not_found", "未找到该自动标注任务。")
        return task

    def stop_task(self, session: Session, task_id: str) -> AutoAnnotationTask:
        task = self.get_task(session, task_id)
        if task.status in {"completed", "failed", "stopped"}:
            return task
        if task.status == "pending":
            task.status = "stopped"
            task.finished_at = utc_now()
            task.error_message = "任务尚未开始即被停止。"
        else:
            task.stop_requested = True
        session.commit()
        session.refresh(task)
        self.queue.stop_pending(task_id)
        return task

    def logs(self, session: Session, task_id: str, tail: int | None = None) -> AutoAnnotationLogResponse:
        task = self.get_task(session, task_id)
        store = TrainingLogStore(task.logs_path or "")
        return AutoAnnotationLogResponse(task_id=task.id, logs=store.read(tail), line_count=store.line_count())

    @staticmethod
    def _resolve_mapping(
        requested: dict[str, str],
        target_classes: list[ClassLabel],
        source_classes: list[ClassLabel],
    ) -> dict[str, str]:
        target_by_id = {item.id: item for item in target_classes}
        target_by_name = {item.name.casefold(): item for item in target_classes}
        mapping: dict[str, str] = {}
        for source in source_classes:
            target = target_by_name.get(source.name.casefold())
            if target is not None:
                mapping[str(source.class_index)] = target.id
        for raw_index, target_id in requested.items():
            try:
                source_index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValidationError("invalid_class_mapping", "模型类别索引必须是整数。") from exc
            if source_index < 0 or target_id not in target_by_id:
                raise ValidationError("invalid_class_mapping", "类别映射必须指向数据集中已有的类别。")
            if source_classes and source_index not in {item.class_index for item in source_classes}:
                raise ValidationError("invalid_class_mapping", "类别映射引用了所选模型未提供的类别。")
            mapping[str(source_index)] = target_id
        if not mapping:
            raise ValidationError("class_mapping_empty", "没有可与该数据集匹配的模型类别。")
        return mapping


def save_auto_annotations(
    session: Session,
    task: AutoAnnotationTask,
    image: ImageItem,
    detections: list[Detection],
) -> tuple[int, int, bool]:
    """Persist model output as editable auto annotations, preserving human labels by default."""

    if task.skip_annotated_images and session.scalar(select(Annotation.id).where(Annotation.image_id == image.id).limit(1)) is not None:
        return 0, 0, True
    if task.clean_old_annotations:
        session.execute(delete(Annotation).where(Annotation.image_id == image.id))
    else:
        session.execute(delete(Annotation).where(Annotation.image_id == image.id, Annotation.source == "auto"))

    created: list[Annotation] = []
    skipped = 0
    expected_type = annotation_type_for(task.task_type)
    for detection in detections:
        class_id = task.class_mapping.get(str(detection.class_index))
        if class_id is None:
            skipped += 1
            continue
        annotation = _annotation_from_detection(task, image, detection, class_id, expected_type)
        if annotation is None:
            skipped += 1
            continue
        created.append(annotation)
    session.add_all(created)
    session.flush()
    annotation_count = session.scalar(select(func.count()).select_from(Annotation).where(Annotation.image_id == image.id)) or 0
    image.status = "annotated" if annotation_count else "unannotated"
    session.commit()
    return len(created), skipped, False


def _annotation_from_detection(
    task: AutoAnnotationTask,
    image: ImageItem,
    detection: Detection,
    class_id: str,
    expected_type: str,
) -> Annotation | None:
    common = {
        "id": new_id("ann"),
        "image_id": image.id,
        "dataset_id": image.dataset_id,
        "class_id": class_id,
        "type": expected_type,
        "source": "auto",
    }
    if expected_type == "classify":
        return Annotation(**common)
    if expected_type == "bbox":
        from app.core.schemas import BBox

        bbox = BBox(x=detection.x, y=detection.y, width=detection.width, height=detection.height)
        validate_bbox(bbox, image.width, image.height)
        return Annotation(**common, x=bbox.x, y=bbox.y, width=bbox.width, height=bbox.height)
    if expected_type == "polygon":
        points = [list(point) for point in detection.polygon or ()]
        if len(points) < 3:
            return None
        polygon = [(float(point[0]), float(point[1])) for point in points]
        validate_polygon(polygon, image.width, image.height)
        return Annotation(**common, polygon=points)
    obb = _obb_from_points(detection.obb_points)
    if obb is None:
        return None
    validate_obb(obb, image.width, image.height)
    return Annotation(**common, obb=obb.model_dump())


def _obb_from_points(points: tuple[tuple[float, float], ...] | None):
    from app.core.schemas import OBB

    if points is None or len(points) != 4:
        return None
    edge_a = math.dist(points[0], points[1])
    edge_b = math.dist(points[1], points[2])
    if edge_a <= 0 or edge_b <= 0:
        return None
    cx = sum(point[0] for point in points) / 4
    cy = sum(point[1] for point in points) / 4
    angle = math.degrees(math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
    return OBB(cx=cx, cy=cy, width=edge_a, height=edge_b, angle=angle)
