from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.auto_annotation.service import save_auto_annotations
from app.core.models import AutoAnnotationTask, ClassLabel, ImageItem, ModelVersion
from app.core.storage import Storage
from app.core.time import utc_now
from app.core.local_compute import local_compute_gate
from app.models.inference import run_managed_inference
from app.training.observability.log_store import TrainingLogStore


class AutoAnnotationRunner:
    def __init__(self, session_factory: sessionmaker[Session], queue, storage: Storage) -> None:
        self.session_factory = session_factory
        self.queue = queue
        self.storage = storage

    def run(self, task_id: str) -> None:
        try:
            spec = self._claim(task_id)
            if spec is None:
                return
            log_store = TrainingLogStore(spec["logs_path"])
            log_store.append(f"自动标注任务 {task_id} 正在等待本地计算资源。")
            with local_compute_gate.acquire():
                if self._stop_requested(task_id):
                    self._finish(task_id, log_store, [])
                    return
                log_store.append(f"自动标注任务 {task_id} 已开始。")
                log_store.append(f"模型：{spec['model_id']} · 共 {spec['total_images']} 张图 · 置信度={spec['confidence']:.2f} · IoU={spec['iou']:.2f}")
                errors: list[str] = []
                for index, image_id in enumerate(spec["image_ids"], start=1):
                    if self._stop_requested(task_id):
                        break
                    try:
                        image_path = self._image_path(spec["dataset_id"], image_id)
                        detections = run_managed_inference(
                            model_id=spec["model_id"],
                            model_path=Path(spec["model_path"]),
                            task_type=spec["task_type"],
                            image_path=image_path,
                            confidence=spec["confidence"],
                            iou=spec["iou"],
                            class_names=spec["class_names"],
                        )
                        detections = [item for item in detections if item.confidence >= spec["confidence"]]
                        if spec["task_type"] == "classify":
                            detections = detections[:1]
                        with self.session_factory() as session:
                            task = session.get(AutoAnnotationTask, task_id)
                            image = session.get(ImageItem, image_id)
                            if task is None or image is None:
                                raise RuntimeError("运行中任务或图片已被删除。")
                            created, skipped, image_skipped = save_auto_annotations(session, task, image, detections)
                            task.processed_images = index
                            task.created_annotations += created
                            task.skipped_images += int(image_skipped)
                            task.progress_percent = round(index / max(task.total_images, 1) * 100, 2)
                            session.commit()
                        if image_skipped:
                            log_store.append(f"{image_id}：入队后该图片已被标注，本次跳过。")
                        if skipped:
                            log_store.append(f"{image_id}：新增 {created} 个，跳过 {skipped} 个无法映射或无效的检测。")
                    except Exception as exc:
                        errors.append(f"{image_id}: {exc}")
                        self._mark_image_failed(task_id, index)
                        log_store.append(errors[-1])
                self._finish(task_id, log_store, errors)
        finally:
            self.queue.on_finished(task_id)

    def _claim(self, task_id: str) -> dict | None:
        with self.session_factory() as session:
            task = session.get(AutoAnnotationTask, task_id)
            if task is None or task.status != "pending":
                return None
            if task.stop_requested:
                task.status = "stopped"
                task.finished_at = utc_now()
                task.error_message = "任务尚未开始即被停止。"
                session.commit()
                return None
            model = session.get(ModelVersion, task.model_id)
            if model is None:
                task.status = "failed"
                task.finished_at = utc_now()
                task.error_message = "任务开始前所选模型已被删除。"
                session.commit()
                return None
            image_query = select(ImageItem.id).where(ImageItem.dataset_id == task.dataset_id)
            if task.skip_annotated_images:
                image_query = image_query.where(~ImageItem.annotations.any())
            image_ids = list(session.scalars(image_query.order_by(ImageItem.created_at, ImageItem.id)))
            source_classes = list(session.scalars(select(ClassLabel).where(ClassLabel.dataset_id == model.dataset_id).order_by(ClassLabel.class_index))) if model.dataset_id else []
            task.status = "running"
            task.started_at = utc_now()
            task.total_images = len(image_ids)
            task.processed_images = 0
            task.progress_percent = 0.0
            session.commit()
            return {
                "dataset_id": task.dataset_id,
                "model_id": task.model_id,
                "model_path": str(self.storage.managed_model_path(model.model_path)),
                "task_type": task.task_type,
                "confidence": task.confidence,
                "iou": task.iou,
                "class_names": {item.class_index: item.name for item in source_classes},
                "image_ids": image_ids,
                "total_images": len(image_ids),
                "logs_path": task.logs_path or str(self.storage.auto_annotation_task_dir(task.id) / "auto_annotation.log"),
            }

    def _image_path(self, dataset_id: str, image_id: str) -> Path:
        with self.session_factory() as session:
            image = session.get(ImageItem, image_id)
            if image is None or image.dataset_id != dataset_id:
                raise RuntimeError("所选数据集中找不到该图片。")
            path = self.storage.image_path(dataset_id, image.storage_name)
        if not path.is_file():
            raise RuntimeError("受管图片文件已丢失。")
        return path

    def _stop_requested(self, task_id: str) -> bool:
        with self.session_factory() as session:
            task = session.get(AutoAnnotationTask, task_id)
            return bool(task and task.stop_requested)

    def _mark_image_failed(self, task_id: str, index: int) -> None:
        with self.session_factory() as session:
            task = session.get(AutoAnnotationTask, task_id)
            if task is None:
                return
            task.processed_images = index
            task.skipped_images += 1
            task.progress_percent = round(index / max(task.total_images, 1) * 100, 2)
            session.commit()

    def _finish(self, task_id: str, log_store: TrainingLogStore, errors: list[str]) -> None:
        with self.session_factory() as session:
            task = session.get(AutoAnnotationTask, task_id)
            if task is None:
                return
            stopped = bool(task.stop_requested)
            task.finished_at = utc_now()
            if stopped:
                task.status = "stopped"
                task.error_message = "用户已停止自动标注。"
                log_store.append(task.error_message)
            elif errors and task.processed_images <= len(errors):
                task.status = "failed"
                task.error_message = f"{len(errors)} 张图片自动标注失败。"
                log_store.append(task.error_message)
            else:
                task.status = "completed"
                task.progress_percent = 100.0
                task.error_message = f"已完成，跳过 {len(errors)} 张图片。" if errors else None
                log_store.append("自动标注已完成。训练前请先审核并修正生成的标注。")
            session.commit()
