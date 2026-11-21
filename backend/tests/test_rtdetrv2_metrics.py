from __future__ import annotations

from pathlib import Path

from app.engines.rtdetrv2_runner import _write_results_csv

# 真实日志片段：同一段里既有 area=all 也有 area=small/medium/large（large 常是 -1）
LOG_SAMPLE = """
Epoch: [0] Total time: 0:00:45 (9.1 s / it)
Test:  [0/1]  eta: 0:00:11    time: 11.7  data: 8.8
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.0942
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.3398
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.1017
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.0000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.0330
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Epoch: [0]  [4/5]  lr: 0.000002  loss: 19.6012 (20.0380)
"""


def test_results_csv_uses_area_all_and_ignores_large(tmp_path: Path) -> None:
    _write_results_csv(tmp_path, 10, LOG_SAMPLE)

    text = (tmp_path / "results.csv").read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert lines[0] == "epoch,time,train/loss,metrics/mAP50(B),metrics/mAP50-95(B)"
    row = lines[1].split(",")
    assert row[0] == "10"
    assert row[2] == "20.0380"
    assert row[3] == "0.3398"  # mAP50
    assert row[4] == "0.0942"  # mAP50-95（不是 -1）


def test_results_csv_skipped_when_no_metrics(tmp_path: Path) -> None:
    _write_results_csv(tmp_path, 1, "Epoch: [0] Total time: 0:00:45\n")
    assert not (tmp_path / "results.csv").exists()
