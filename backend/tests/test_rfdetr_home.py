from __future__ import annotations

import os

from app.engines import rfdetr_home


def test_rf_home_defaults_inside_project(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(rfdetr_home.ENV_KEY, raising=False)
    monkeypatch.setattr(rfdetr_home, "default_home", lambda: tmp_path / "weights" / "rfdetr")

    path = rfdetr_home.ensure_rf_home()

    assert path == tmp_path / "weights" / "rfdetr"
    assert path.is_dir()
    assert os.environ[rfdetr_home.ENV_KEY] == str(path)


def test_rf_home_respects_explicit_setting(monkeypatch, tmp_path) -> None:
    custom = tmp_path / "custom"
    monkeypatch.setenv(rfdetr_home.ENV_KEY, str(custom))

    path = rfdetr_home.ensure_rf_home()

    assert path == custom
    assert path.is_dir()
