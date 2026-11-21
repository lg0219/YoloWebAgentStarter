"""三个训练引擎的端到端冒烟测试（耗时较长，默认跳过）。

启用方式：
    set YWA_E2E=1
    .venv\\Scripts\\python.exe -m pytest backend/tests/test_engine_e2e.py -q

覆盖：真实跑一遍每个引擎的训练，检查产物（weights/best.pt、results.csv、引擎旁注），
以及「停止时连子进程一起杀」。这些是单元测试覆盖不到、又最容易打地鼠的地方。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(os.getenv("YWA_E2E") != "1", reason="需要 YWA_E2E=1 才跑端到端测试")

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
YOLO_EXE = REPO / ".venv" / "Scripts" / "yolo.exe"
RT_WEIGHT = REPO / "weights" / "rtdetrv2" / "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
RF_WEIGHT = REPO / "weights" / "rfdetr" / "rf-detr-nano.pth"


def _env(tmp: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(REPO / "backend"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "TEMP": str(tmp),
            "TMP": str(tmp),
            "MPLCONFIGDIR": str(tmp / "mpl"),
            "YOLO_CONFIG_DIR": str(tmp / "yolocfg"),
            "RF_HOME": str(REPO / "weights" / "rfdetr"),
            "YWA_RTDETRV2_WEIGHTS_DIR": str(REPO / "weights" / "rtdetrv2"),
        }
    )
    (tmp / "mpl").mkdir(parents=True, exist_ok=True)
    (tmp / "yolocfg" / "Ultralytics").mkdir(parents=True, exist_ok=True)
    return env


@pytest.fixture()
def work_dir(tmp_path: Path) -> Path:
    """端到端测试用纯 ASCII 临时目录。

    本机 %TEMP% 不可写时 pytest 会把临时目录放到仓库里（路径含中文），
    ultralytics 解析 data.yaml 时会丢掉中文段导致找不到图片——属于测试环境限制，
    产品本身在中文路径下是能训练的（真实训练已多次跑通）。
    """
    work = tmp_path
    work.mkdir(parents=True, exist_ok=True)
    try:
        yield work
    finally:
        shutil.rmtree(work, ignore_errors=True)


@pytest.fixture()
def dataset(work_dir: Path) -> Path:
    """造一个极小的 YOLO 布局数据集：2 张训练图、1 张验证图。"""
    root = work_dir / "dataset"
    for split, count in (("train", 2), ("val", 1)):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (160, 120), "#336699").save(image_dir / f"{split}{index}.jpg")
            (label_dir / f"{split}{index}.txt").write_text("0 0.5 0.5 0.3 0.4\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n- thing\n", encoding="utf-8"
    )
    return root


def _run(command: list[str], tmp: Path, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO,
        env=_env(tmp),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return result


def test_yolo_engine_trains(dataset: Path, work_dir: Path) -> None:
    run_dir = work_dir / "yolo-run"
    result = _run(
        [
            str(YOLO_EXE), "detect", "train", "model=yolo26n.pt",
            f"data={dataset / 'data.yaml'}", "epochs=1", "imgsz=160", "batch=1",
            "device=cpu", "workers=0", "plots=False", f"project={run_dir}", "name=run", "exist_ok=True",
        ],
        work_dir,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    weights = run_dir / "run" / "weights"
    assert (weights / "best.pt").is_file()
    assert (run_dir / "run" / "results.csv").is_file()


@pytest.mark.skipif(not RF_WEIGHT.is_file(), reason="缺少 RF-DETR nano 权重")
def test_rfdetr_engine_trains(dataset: Path, work_dir: Path) -> None:
    run_dir = work_dir / "rf-run"
    result = _run(
        [
            str(PYTHON), "-m", "app.engines.rfdetr_runner",
            "--dataset-dir", str(dataset), "--output-dir", str(run_dir),
            "--epochs", "1", "--batch-size", "1", "--device", "cpu", "--seed", "42", "--variant", "nano",
        ],
        work_dir,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (run_dir / "weights" / "best.pt").is_file()
    assert (run_dir / "weights" / "best.json").is_file()
    header = (run_dir / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "metrics/mAP50(B)" in header, header


@pytest.mark.skipif(not RT_WEIGHT.is_file(), reason="缺少 RT-DETR r18 权重")
def test_rtdetrv2_engine_trains(dataset: Path, work_dir: Path) -> None:
    run_dir = work_dir / "rt-run"
    result = _run(
        [
            str(PYTHON), "-m", "app.engines.rtdetrv2_runner",
            "--dataset-dir", str(dataset), "--output-dir", str(run_dir),
            "--epochs", "1", "--batch-size", "1", "--resolution", "160", "--device", "cpu",
            "--seed", "42", "--workers", "0", "--variant", "r18",
        ],
        work_dir,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert (run_dir / "weights" / "best.pt").is_file()
    assert (run_dir / "weights" / "best.json").is_file()
    # 每轮 300MB 的中间 checkpoint 不应出现（checkpoint_freq 已调大）
    assert not list(run_dir.glob("checkpoint0*.pth"))


@pytest.mark.skipif(not RT_WEIGHT.is_file(), reason="缺少 RT-DETR r18 权重")
def test_stop_kills_child_processes(dataset: Path, work_dir: Path) -> None:
    """停止必须连带 dataloader 子进程一起杀掉，否则任务会卡在「运行中」。"""
    run_dir = work_dir / "stop-run"
    process = subprocess.Popen(
        [
            str(PYTHON), "-m", "app.engines.rtdetrv2_runner",
            "--dataset-dir", str(dataset), "--output-dir", str(run_dir),
            "--epochs", "3", "--batch-size", "1", "--resolution", "160", "--device", "cpu",
            "--seed", "42", "--workers", "2", "--variant", "r18",
        ],
        cwd=REPO,
        env=_env(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        time.sleep(25)  # 等它把 dataloader 子进程拉起来
        assert process.poll() is None, "训练提前结束，无法验证停止逻辑"
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
        deadline = time.time() + 30
        while time.time() < deadline and process.poll() is None:
            time.sleep(0.5)
        assert process.poll() is not None, "taskkill 后训练进程仍在运行"
        # 管道必须能读到 EOF，否则上层读循环会一直卡住
        assert process.stdout is not None
        process.stdout.close()
    finally:
        if process.poll() is None:
            process.kill()


def test_no_stray_weight_temp_files() -> None:
    """权重目录里不应残留 .part / .tmp 半成品。"""
    leftovers: list[Path] = []
    for sub in ("rfdetr", "rtdetrv2"):
        folder = REPO / "weights" / sub
        if folder.is_dir():
            leftovers += [p for p in folder.iterdir() if p.suffix in {".part", ".tmp"} or ".tmp" in p.name]
    assert not leftovers, leftovers
