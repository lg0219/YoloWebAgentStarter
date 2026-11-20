from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.errors import ValidationError
from app.core.task_types import TaskType
from app.engines.base import TrainSpec
from app.engines.registry import (
    available_engines,
    configured_engine_key,
    engine_for_weight,
    get_engine,
)


def _spec(**overrides) -> TrainSpec:
    base = dict(
        task_type=TaskType.DETECT,
        model="yolo11n.pt",
        data=str(Path("dataset") / "data.yaml"),
        run_dir=str(Path("run")),
        epochs=3,
        img_size=640,
        batch_size=2,
        device="cpu",
        workers=0,
        seed=42,
    )
    base.update(overrides)
    return TrainSpec(**base)


def test_default_engine_is_yolo() -> None:
    engine = get_engine()
    assert engine.info.key == "yolo"
    assert engine.info.display_name == "Ultralytics YOLO"


def test_registry_lists_yolo_and_rfdetr() -> None:
    keys = {info.key for info in available_engines()}
    assert {"yolo", "rfdetr"} <= keys


def test_yolo_supports_community_tasks() -> None:
    engine = get_engine("yolo")
    for task in (TaskType.DETECT, TaskType.SEGMENT, TaskType.OBB, TaskType.CLASSIFY):
        assert engine.supports_task(task)


def test_rfdetr_only_supports_detect() -> None:
    engine = get_engine("rfdetr")
    assert engine.supports_task(TaskType.DETECT)
    assert not engine.supports_task(TaskType.SEGMENT)


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValidationError):
        get_engine("rtdetr")


def test_engine_selected_by_weight_suffix() -> None:
    assert engine_for_weight("model.pt").info.key == "yolo"
    assert engine_for_weight("model.pth").info.key == "rfdetr"
    assert engine_for_weight("model.unknown", fallback="yolo").info.key == "yolo"


def test_command_prefix_is_non_empty_argv() -> None:
    for key in ("yolo", "rfdetr"):
        prefix = get_engine(key).command_prefix()
        assert isinstance(prefix, list)
        assert prefix and all(isinstance(item, str) for item in prefix)


def test_yolo_train_argv_starts_with_engine_key() -> None:
    argv = get_engine("yolo").build_train_argv(_spec())
    assert argv[0] == "yolo"
    assert any(item.startswith("model=") for item in argv)
    assert any(item.startswith("data=") for item in argv)


def test_yolo_train_argv_carries_deterministic_switch() -> None:
    """确定性训练开关要显式落到命令行，方便在命令预览里核对。"""

    on = get_engine("yolo").build_train_argv(_spec(deterministic=True))
    off = get_engine("yolo").build_train_argv(_spec(deterministic=False))
    absent = get_engine("yolo").build_train_argv(_spec())

    assert "deterministic=True" in on
    assert "deterministic=False" in off
    assert not any(item.startswith("deterministic=") for item in absent)


def test_rfdetr_train_argv_uses_runner_flags() -> None:
    argv = get_engine("rfdetr").build_train_argv(
        _spec(model="checkpoints/rfdetr-base.pth", device="cpu")
    )
    assert argv[0] == "rfdetr"
    assert "--dataset-dir" in argv
    assert "--output-dir" in argv
    assert "--weights" in argv


def test_rfdetr_rejects_non_detect_task() -> None:
    with pytest.raises(ValidationError):
        get_engine("rfdetr").build_train_argv(_spec(task_type=TaskType.SEGMENT))


def test_configured_engine_key_reads_settings_json(tmp_path: Path) -> None:
    assert configured_engine_key(tmp_path) == "yolo"
    (tmp_path / "settings.json").write_text(
        json.dumps({"runtime": {"model_engine": "rfdetr"}}), encoding="utf-8"
    )
    assert configured_engine_key(tmp_path) == "rfdetr"


def test_weight_suffix_judgement() -> None:
    yolo = get_engine("yolo")
    assert yolo.accepts_weight("yolo11n.pt")
    assert not yolo.accepts_weight("notes.txt")
    rfdetr = get_engine("rfdetr")
    assert rfdetr.accepts_weight("rfdetr-base.pth")
    assert not rfdetr.accepts_weight("yolo11n.pt")
