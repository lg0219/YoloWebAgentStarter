from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SamFallbackMode = Literal["box", "disabled"]
LLMAuthScheme = Literal["bearer", "header", "raw_authorization"]


class SamSettingsResponse(BaseModel):
    enabled: bool
    model: str
    device: str
    img_size: int
    fallback_mode: SamFallbackMode
    model_configured: bool


class SamSettingsUpdate(BaseModel):
    enabled: bool = True
    model: str = Field(default="", max_length=1024)
    device: str = Field(default="auto", min_length=1, max_length=64)
    img_size: int = Field(default=1024, ge=64, le=4096)
    fallback_mode: SamFallbackMode = "box"


# （lg0219）标注画布偏好：线宽以「屏幕像素」口径保存（zoom 补偿由前端负责），
# 持久化位置与 Sam/LLM 一致（settings.json，不入 SQLite）。
class AnnotationSettingsResponse(BaseModel):
    line_width: float


class AnnotationSettingsUpdate(BaseModel):
    line_width: float = Field(default=1.0, ge=0.5, le=3.0)


# Adapted from upstream YoloWebAgent settings/schemas.py (Community subset).
class LLMSettingsOut(BaseModel):
    enabled: bool = False
    provider: str = "openai-compatible"
    api_base: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    timeout_seconds: int = 60
    api_key_configured: bool = False
    auth_scheme: LLMAuthScheme = "bearer"
    auth_header_name: str = ""


class LLMSettingsUpdate(BaseModel):
    enabled: bool = False
    provider: str = "openai-compatible"
    api_base: str = ""
    api_key: str | None = Field(default=None, description="Leave empty to keep the existing key.")
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=60, ge=1, le=7200)
    auth_scheme: LLMAuthScheme = "bearer"
    auth_header_name: str = ""


class LLMConnectionTestRequest(BaseModel):
    """Test connectivity without persisting settings (upstream pattern)."""

    api_base: str = ""
    model: str = "gpt-4o-mini"
    api_key: str | None = Field(default=None, description="Optional; empty uses saved key.")
    timeout_seconds: int = Field(default=60, ge=1, le=7200)
    auth_scheme: LLMAuthScheme = "bearer"
    auth_header_name: str = ""


class LLMConnectionTestResult(BaseModel):
    ok: bool
    message: str
    remote_test_performed: bool = False
    # Stable machine code for UI localization (message remains Chinese for logs/compat).
    code: str = "ok"
    http_status: int | None = None


class LLMSettingsInternal(LLMSettingsOut):
    api_key: str = ""


# （lg0219）智能分析（vlm-bridge）云端解释配置。
# 只作用于流水线的「解释」阶段；未启用时仍使用本地模型，行为不变。
class PipelineSettingsOut(BaseModel):
    # 默认关闭：未部署 vlm-bridge 时不显示「智能分析」入口，也不产生健康检查请求
    enabled: bool = False
    cloud_enabled: bool = False
    api_base: str = ""
    model: str = ""
    api_key_configured: bool = False
    timeout_seconds: int = 60
    max_tokens: int = 256


class PipelineSettingsUpdate(BaseModel):
    enabled: bool = False
    cloud_enabled: bool = False
    api_base: str = Field(default="", max_length=1024)
    model: str = Field(default="", max_length=256)
    api_key: str | None = Field(default=None, description="留空表示沿用已保存的密钥。")
    timeout_seconds: int = Field(default=60, ge=5, le=600)
    max_tokens: int = Field(default=256, ge=32, le=4096)


# （lg0219）运行参数：保存后立即生效，不影响监听地址等启动参数。
class RuntimeSettingsOut(BaseModel):
    yolo_executable: str = ""
    max_upload_mb: int = 50
    max_video_upload_mb: int = 2048
    max_video_duration_seconds: int = 14400
    # （lg0219）预训练权重下载源：official / ghfast / gh-proxy / ghproxy-net / custom
    model_source: str = "official"
    model_source_custom: str = ""


class RuntimeSettingsUpdate(BaseModel):
    yolo_executable: str = Field(default="", max_length=1024)
    max_upload_mb: int = Field(default=50, ge=1, le=4096)
    max_video_upload_mb: int = Field(default=2048, ge=1, le=102400)
    max_video_duration_seconds: int = Field(default=14400, ge=10, le=86400)
    model_source: str = Field(default="official", max_length=32)
    model_source_custom: str = Field(default="", max_length=512)


# （lg0219）部署信息：只读，来自进程启动时的环境变量，界面仅展示。
class DeploymentItem(BaseModel):
    key: str
    label: str
    value: str
    note: str = ""


class DeploymentInfoResponse(BaseModel):
    items: list[DeploymentItem]
