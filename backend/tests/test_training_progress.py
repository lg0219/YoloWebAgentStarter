from __future__ import annotations

import pytest

from app.training.observability.metrics import TrainingMetricsParser
from app.training.runtime.runner import resolve_progress


@pytest.fixture()
def parser() -> TrainingMetricsParser:
    return TrainingMetricsParser()


def test_yolo_epoch_line_updates_progress(parser) -> None:
    line = "        3/10      1.234      2.345      0.456        16        320:  40% ━━━━━ 4/10 1.2s/it"
    resolved = resolve_progress("yolo", line, 10, parser)
    assert resolved == {"epoch": 3, "total_epochs": 10}


def test_rtdetr_only_counts_finished_epoch(parser) -> None:
    # 该轮跑完：算作已完成 1 轮
    done = "[ywa] Epoch: [0] Total time: 0:00:56 (11.3 s / it)"
    assert resolve_progress("rtdetrv2", done, 10, parser) == {"epoch": 1, "total_epochs": 10}

    # 批次进度：不是轮次，必须忽略（曾把它当成 0/5）
    batch = "Epoch: [1]  [0/5]  eta: 0:01:33  lr: 0.000002  loss: 18.1"
    assert resolve_progress("rtdetrv2", batch, 10, parser) is None

    # 验证进度：曾经把进度覆盖回 0/1
    test_line = "Test:  [0/1]  eta: 0:00:11    time: 11.7  data: 8.8"
    assert resolve_progress("rtdetrv2", test_line, 10, parser) is None


def test_rtdetr_progress_is_monotonic(parser) -> None:
    """逐轮跑完时进度必须递增，不能被后面的噪声行覆盖回去。"""

    epochs = 3
    current: dict[str, int] | None = None
    lines = []
    for epoch in range(epochs):
        lines += [
            f"Epoch: [{epoch}]  [0/5]  eta: 0:01:33  loss: 18.1",
            f"Epoch: [{epoch}]  [4/5]  eta: 0:00:11  loss: 19.6",
            f"Epoch: [{epoch}] Total time: 0:00:56 (11.3 s / it)",
            "Test:  [0/1]  eta: 0:00:11    time: 11.7  data: 8.8",
            " Average Precision  (AP) @[ IoU=0.50 | area= all | maxDets=100 ] = 0.000",
        ]
    seen: list[int] = []
    for line in lines:
        resolved = resolve_progress("rtdetrv2", line, epochs, parser)
        if resolved:
            current = resolved
            seen.append(resolved["epoch"])
    assert seen == [1, 2, 3]
    assert current == {"epoch": 3, "total_epochs": 3}


def test_rtdetr_progress_clamped_to_total(parser) -> None:
    resolved = resolve_progress("rtdetrv2", "Epoch: [99] Total time: 0:00:01", 10, parser)
    assert resolved == {"epoch": 10, "total_epochs": 10}
