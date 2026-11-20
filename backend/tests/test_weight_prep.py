from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import ValidationError
from app.engines import weight_prep


def test_only_engine_backed_frameworks_use_prep() -> None:
    assert weight_prep.supports("rfdetr")
    assert weight_prep.supports("rtdetrv2")
    assert not weight_prep.supports("yolo")


def test_rfdetr_variant_maps_to_official_file() -> None:
    path, filename = weight_prep._rf_target("nano")
    assert filename == "rf-detr-nano.pth"
    assert path.name == filename

    default_path, default_name = weight_prep._rf_target("")
    assert default_name == "rf-detr-base.pth"
    assert default_path.name == default_name

    with pytest.raises(ValidationError):
        weight_prep._rf_target("huge")


def test_rtdetrv2_variant_maps_to_backbone_weight() -> None:
    path, filename = weight_prep._rtdetrv2_target("r50")
    assert filename == "ResNet50_vd_ssld_v2_pretrained_from_paddle.pth"
    assert path.name == filename

    with pytest.raises(ValidationError):
        weight_prep._rtdetrv2_target("r999")


def test_status_reports_missing_weight_without_downloading() -> None:
    info = weight_prep.status("rfdetr", "nano")
    assert info["kind"] == "engine"
    assert info["download"]["status"] in {"idle", "downloading", "ready", "failed"}
    assert Path(info["path"]).name == "rf-detr-nano.pth"
