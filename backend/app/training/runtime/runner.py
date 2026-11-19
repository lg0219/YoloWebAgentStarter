from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.core.models import TrainingTask
from app.core.time import utc_now
from app.core.storage import Storage
from app.core.local_compute import local_compute_gate
from app.training.artifacts.checkpoints import checkpoint_paths
from app.training.observability.log_store import TrainingLogStore
from app.training.observability.metrics import TrainingMetricsParser
from app.training.observability.summary import write_training_summary
from app.training.runtime.process_registry import process_registry


def _configured_yolo_executable() -> str:
    """读取设置页保存的 YOLO 可执行文件路径（settings.json 的 runtime 段）。"""
    try:
        from app.core.config import Settings

        path = Settings.from_env().data_dir / "settings.json"
        if not path.is_file():
            return ""
        data = json.loads(path.read_text(encoding="utf-8"))
        value = (data.get("runtime") or {}).get("yolo_executable") or ""
        return str(value).strip()
    except Exception:
        # 设置文件损坏或不可读时退回默认解析，不影响训练
        return ""


# （lg0219）预训练权重下载源：镜像只是加速通道，文件仍来自官方仓库
_MODEL_MIRRORS = {
    "ghfast": "https://ghfast.top",
    "gh-proxy": "https://gh-proxy.com",
    "ghproxy-net": "https://ghproxy.net",
}
_OFFICIAL_ASSET_BASE = "https://github.com/ultralytics/assets/releases/download/v0.0.0"


def _read_runtime_section(data_dir: Path) -> dict:
    try:
        path = data_dir / "settings.json"
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("runtime") or {}
    except Exception:
        return {}


def apply_model_source(args: list[str], data_dir: Path) -> list[str]:
    """按设置把裸权重名改写成镜像 URL，交由 Ultralytics 从该地址下载。

    仅作用于 `model=xxx.pt` 的裸文件名；本地路径、已有 URL、.yaml 一律原样保留。
    """
    runtime = _read_runtime_section(data_dir)
    source = str(runtime.get("model_source") or "official").strip()
    if source in ("", "official"):
        return args
    if source == "custom":
        prefix = str(runtime.get("model_source_custom") or "").strip()
    else:
        prefix = _MODEL_MIRRORS.get(source, "")
    if not prefix:
        return args
    prefix = prefix.rstrip("/")
    rewritten: list[str] = []
    for arg in args:
        if arg.startswith("model=") and arg.endswith(".pt"):
            name = arg[len("model="):]
            if not name.startswith(("http://", "https://")) and "/" not in name and "\\" not in name:
                rewritten.append(f"model={prefix}/{_OFFICIAL_ASSET_BASE}/{name}")
                continue
        rewritten.append(arg)
    return rewritten


def resolve_yolo_command_prefix() -> list[str]:
    override = os.getenv("YWA_YOLO_EXECUTABLE", "").strip()
    if override:
        return [override]
    configured = _configured_yolo_executable()
    if configured and Path(configured).expanduser().exists():
        return [str(Path(configured).expanduser())]
    repository_root = Path(__file__).resolve().parents[4]
    candidate = repository_root / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("yolo.exe" if os.name == "nt" else "yolo")
    if candidate.is_file():
        return [str(candidate)]
    found = shutil.which("yolo")
    if found:
        return [found]
    return [sys.executable, "-c", "from ultralytics.cfg import entrypoint; raise SystemExit(entrypoint())"]


class TrainingRunner:
    def __init__(self, session_factory: sessionmaker[Session], queue, storage: Storage | None = None) -> None:
        self.session_factory = session_factory
        self.queue = queue
        self.storage = storage
        self.metrics_parser = TrainingMetricsParser()

    def run(self, task_id: str) -> None:
        process: subprocess.Popen[str] | None = None
        try:
            task = self._claim(task_id)
            if task is None:
                return
            log_store = TrainingLogStore(task.logs_path or "train.log")
            log_store.append(f"Training task {task.id} is waiting for the local compute slot.")
            with local_compute_gate.acquire():
                if self._stop_requested(task_id):
                    self._finish(task_id, -1, log_store)
                    return
                log_store.append(f"Training started for task {task.id}.")
                command = self._command(task)
                log_store.append("Command: " + " ".join(command))
                try:
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                        shell=False,
                        start_new_session=os.name == "posix",
                    )
                    process_registry.register(task_id, process)
                    if self._stop_requested(task_id):
                        process_registry.request_stop(task_id)
                    assert process.stdout is not None
                    for line in process.stdout:
                        clean_line = line.rstrip("\n")
                        log_store.append(clean_line)
                        self._update_progress(task_id, clean_line)
                    return_code = process.wait()
                except Exception as exc:
                    log_store.append(f"Training process failed to start or read output: {exc}")
                    self._finish(task_id, -1, log_store, process_error=str(exc))
                    return
                self._finish(task_id, return_code, log_store)
        finally:
            process_registry.unregister(task_id)
            self.queue.on_finished(task_id)

    def _claim(self, task_id: str) -> TrainingTask | None:
        with self.session_factory() as session:
            task = session.get(TrainingTask, task_id)
            if task is None or task.status != "pending":
                return None
            if task.stop_requested:
                task.status = "stopped"
                task.finished_at = utc_now()
                task.error_message = "Training stopped before the process started."
                session.commit()
                return None
            task.status = "running"
            task.started_at = utc_now()
            task.progress_total_epochs = task.epochs
            task.progress_epoch = 0
            task.progress_percent = 0.0
            session.commit()
            session.refresh(task)
            return task

    def _command(self, task: TrainingTask) -> list[str]:
        args = [str(item) for item in (task.command_args_json or [])]
        if not args:
            raise RuntimeError("Training command is missing.")
        if args[0] == "yolo":
            command = [*resolve_yolo_command_prefix(), *args[1:]]
            data_dir = self.storage.data_dir if self.storage else Path(".")
            return apply_model_source(command, data_dir)
        return args

    def _stop_requested(self, task_id: str) -> bool:
        if process_registry.stop_requested(task_id):
            return True
        with self.session_factory() as session:
            task = session.get(TrainingTask, task_id)
            return bool(task and task.stop_requested)

    def _update_progress(self, task_id: str, line: str) -> None:
        progress = self.metrics_parser.parse_progress_line(line)
        if not progress:
            return
        with self.session_factory() as session:
            task = session.get(TrainingTask, task_id)
            if task is None or task.status != "running":
                return
            if "epoch" in progress:
                total = int(progress.get("total_epochs", task.epochs))
                epoch = min(int(progress.get("epoch", 0)), total)
                task.progress_total_epochs = max(total, 1)
                task.progress_epoch = max(epoch, 0)
                task.progress_percent = round(task.progress_epoch / task.progress_total_epochs * 100, 2)
            live_metrics = dict(task.metrics_json or {})
            for key in ("batch_time_seconds", "speed_it_per_sec"):
                value = progress.get(key)
                if value is not None:
                    live_metrics[key] = value
            if live_metrics != (task.metrics_json or {}):
                task.metrics_json = live_metrics
            session.commit()

    def _finish(self, task_id: str, return_code: int, log_store: TrainingLogStore, process_error: str | None = None) -> None:
        with self.session_factory() as session:
            task = session.get(TrainingTask, task_id)
            if task is None:
                return
            stopped = self._stop_requested(task_id)
            artifacts = checkpoint_paths(task.run_dir or "")
            task.best_model_path = artifacts["best"]
            task.last_model_path = artifacts["last"]
            if task.run_dir:
                final_metrics = self.metrics_parser.parse_results(Path(task.run_dir) / "results.csv")
                task.metrics_json = {**dict(task.metrics_json or {}), **final_metrics}
            task.finished_at = utc_now()
            if stopped:
                task.status = "stopped"
                task.error_message = "Training stopped by user."
                log_store.append(task.error_message)
            elif return_code == 0 and artifacts["best"] and artifacts["last"]:
                task.status = "completed"
                task.progress_epoch = task.progress_total_epochs or task.epochs
                task.progress_total_epochs = task.progress_total_epochs or task.epochs
                task.progress_percent = 100.0
                task.error_message = None
                log_store.append("Training completed with best.pt and last.pt.")
            else:
                task.status = "failed"
                task.error_message = process_error or (
                    "Training completed without both best.pt and last.pt."
                    if return_code == 0
                    else f"YOLO training exited with code {return_code}."
                )
                log_store.append(task.error_message)
            if task.status == "completed" and self.storage is not None:
                completion_state = {
                    "best_model_path": task.best_model_path,
                    "last_model_path": task.last_model_path,
                    "metrics_json": dict(task.metrics_json or {}),
                    "finished_at": task.finished_at,
                    "progress_epoch": task.progress_epoch,
                    "progress_total_epochs": task.progress_total_epochs,
                    "progress_percent": task.progress_percent,
                }
                try:
                    from app.models.service import ModelService

                    registered = ModelService(self.storage).register_training_artifacts(session, task)
                    log_store.append(f"Registered {len(registered)} model artifacts.")
                except Exception as exc:
                    task = session.get(TrainingTask, task_id)
                    if task is None:
                        return
                    for field, value in completion_state.items():
                        setattr(task, field, value)
                    task.status = "failed"
                    task.error_message = f"Training artifacts could not be registered: {exc}"
                    log_store.append(task.error_message)
            dataset = task.dataset
            write_training_summary(task, dataset)
            session.commit()
