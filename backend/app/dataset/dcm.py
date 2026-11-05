"""（lg0219）DICOM（.dcm）导入支持：导入阶段转成 8bit PNG。

背景：工业 DR 设备输出 16-bit **单帧** DICOM，同目录放同名 `.JSON`（采集与显示参数）
和同名 `.BMP`（设备自己渲染的 8bit 预览）。标注/训练/导出整条链路只认 8bit 图片，
所以在「导入」这一步把 DICOM 转成 PNG，后续代码完全不感知 DICOM。

🔴 关于窗宽窗位（基于实测，勿照抄 JSON）：
    JSON 的 `view.params.tfm.ww_val / wl_val`（样例 21570 / 44944）**不能**直接套到
    DICOM 原始像素值上。样例原始值域约 10293~23093，而该窗口区间是
    [44944-10785, 44944+10785] = [34159, 55729]，与值域完全不重叠，
    直接套用输出全黑（实测输出均值 0.0）。

    设备实际渲染用的是一条**固定色调曲线**。用设备自带 .BMP 预览逐像素反推得到：

        L = 255 * clip((raw - LOW) / (HIGH - LOW), 0, 1) ** GAMMA

    参数 LOW=9800 / HIGH=27100 / GAMMA=0.42 由两张不同片位、不同值域的图像交叉验证：
        A1-1（值域 10864~23093）：RMSE 2.7 灰度级
        A2-1（值域 10293~21156）：RMSE 3.0 灰度级
    视觉与设备预览一致。换设备/换工艺时可用环境变量重新校正，无需改代码：
        YWA_DCM_DISPLAY_LOW / YWA_DCM_DISPLAY_HIGH / YWA_DCM_DISPLAY_GAMMA
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DICOM_SUFFIXES = {".dcm", ".dicom"}

# 设备色调曲线默认参数（见模块注释的实测依据）。
_DEFAULT_DISPLAY_LOW = 9800.0
_DEFAULT_DISPLAY_HIGH = 27100.0
_DEFAULT_DISPLAY_GAMMA = 0.42

# DICOM Part 10 文件在偏移 128 处有 "DICM" 魔数。
_DICM_MAGIC = b"DICM"
_DICM_OFFSET = 128


def is_dicom_name(name: str) -> bool:
    """按扩展名判断（大小写不敏感）。"""

    return Path(name).suffix.lower() in DICOM_SUFFIXES


def looks_like_dicom(content: bytes) -> bool:
    """按文件内容判断，用于扩展名不规范但仍需处理的场景。"""

    return len(content) >= _DICM_OFFSET + len(_DICM_MAGIC) and content[_DICM_OFFSET : _DICM_OFFSET + len(_DICM_MAGIC)] == _DICM_MAGIC


def find_sidecar_json(dicom_path: Path) -> Path | None:
    """找同名 JSON 侧车文件（设备导出大小写不统一，逐个试）。"""

    for suffix in (".JSON", ".json", ".Json"):
        candidate = dicom_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _display_params() -> tuple[float, float, float]:
    """允许用环境变量覆盖曲线参数，便于换设备时校正。"""

    def _read(name: str, fallback: float) -> float:
        try:
            return float(os.getenv(name, "") or fallback)
        except ValueError:
            return fallback

    low = _read("YWA_DCM_DISPLAY_LOW", _DEFAULT_DISPLAY_LOW)
    high = _read("YWA_DCM_DISPLAY_HIGH", _DEFAULT_DISPLAY_HIGH)
    gamma = _read("YWA_DCM_DISPLAY_GAMMA", _DEFAULT_DISPLAY_GAMMA)
    if not high > low or gamma <= 0:
        return _DEFAULT_DISPLAY_LOW, _DEFAULT_DISPLAY_HIGH, _DEFAULT_DISPLAY_GAMMA
    return low, high, gamma


@dataclass(frozen=True)
class DicomConversion:
    """转换结果：PNG 字节 + 供日志/后续功能使用的元数据。"""

    png: bytes
    width: int
    height: int
    metadata: dict[str, Any]


def read_sidecar_metadata(json_bytes: bytes | None) -> dict[str, Any]:
    """读取侧车 JSON 里**仅作记录**的信息。

    注意：ww_val / wl_val 只是记录下来，不参与转换（原因见模块注释）。
    """

    if not json_bytes:
        return {}
    try:
        data = json.loads(json_bytes)
    except (ValueError, TypeError):
        return {"sidecar_error": "invalid_json"}
    view = data.get("view") or {}
    params = view.get("params") or {}
    tfm = params.get("tfm") or {}
    acquisition = data.get("acquisition") or {}
    interpretation = data.get("interpretation") or {}
    detector = (acquisition.get("detector") or {}).get("config") or {}
    return {
        # 记录下来备查，但**不用于**转换（设备值不是原始像素单位）
        "json_ww": tfm.get("ww_val"),
        "json_wl": tfm.get("wl_val"),
        "watermark": (view.get("overlays") or {}).get("watermark"),
        "reviewed": interpretation.get("is_reviewed"),
        "defect_level": interpretation.get("defect_level"),
        "pixel_size": detector.get("pixel_size"),
        "frame_rate": detector.get("frame_rate"),
    }


def convert_dicom_to_png(content: bytes, sidecar: bytes | None = None) -> DicomConversion:
    """把单帧 DICOM 转成 8bit 灰度 PNG。

    抛出 `ValidationError`（由调用方转成 4xx），不抛裸异常。
    """

    # 延迟导入：不导入 DICOM 的进程（如只跑测试的后端子集）无需加载 pydicom/numpy。
    import numpy as np
    import pydicom
    from PIL import Image

    from app.core.errors import ValidationError

    if not (is_dicom_name("x.dcm") or looks_like_dicom(content)):  # pragma: no cover - 调用方已判断
        raise ValidationError("invalid_dicom", "File is not a DICOM image.")

    try:
        dataset = pydicom.dcmread(io.BytesIO(content), force=False)
    except Exception as exc:  # pydicom 会抛各种解析异常，统一收口
        raise ValidationError("invalid_dicom", "无法解析 DICOM 文件。") from exc

    frames = _frame_count(dataset)
    if frames > 1:
        # 多帧序列导入策略未定（是否拆帧、帧序与标注如何对应），先明确拒绝而不是静默丢帧。
        raise ValidationError(
            "unsupported_dicom_multiframe",
            f"暂不支持多帧 DICOM（检测到 {frames} 帧）。请先拆成单帧再导入。",
        )

    try:
        pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    except Exception as exc:
        raise ValidationError("invalid_dicom_pixels", "DICOM 像素数据无法解码。") from exc
    if pixels.ndim != 2:
        raise ValidationError("invalid_dicom_pixels", "DICOM 像素不是二维灰度图像。")

    # 设备可能用 RescaleSlope/Intercept 把存储值映射到物理值；有就应用，没有就原样。
    slope = _as_float(getattr(dataset, "RescaleSlope", None), 1.0)
    intercept = _as_float(getattr(dataset, "RescaleIntercept", None), 0.0)
    if slope not in (0.0, 1.0) or intercept:
        pixels = pixels * slope + intercept

    low, high, gamma = _display_params()
    if _as_int(getattr(dataset, "PixelRepresentation", 0), 0) == 1:
        # 有符号数据：先平移到非负，避免曲线参数口径错位（工业 DR 一般是 0）。
        pixels = pixels - float(pixels.min())

    normalized = np.clip((pixels - low) / (high - low), 0.0, 1.0)
    if gamma != 1.0:
        normalized = np.power(normalized, gamma)
    eight_bit = np.rint(normalized * 255.0).astype(np.uint8)

    image = Image.fromarray(eight_bit, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)

    metadata = read_sidecar_metadata(sidecar)
    metadata.update(
        {
            "raw_min": float(pixels.min()),
            "raw_max": float(pixels.max()),
            "display_low": low,
            "display_high": high,
            "display_gamma": gamma,
        }
    )
    height, width = eight_bit.shape
    return DicomConversion(png=buffer.getvalue(), width=int(width), height=int(height), metadata=metadata)


def _frame_count(dataset: Any) -> int:
    raw = getattr(dataset, "NumberOfFrames", None)
    count = _as_int(raw, 1)
    return count if count > 0 else 1


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
