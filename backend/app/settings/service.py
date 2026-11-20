from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from typing import Any

import httpx

from app.agent.llm_auth import (
    build_llm_request_headers,
    coalesce_auth_scheme,
    validate_auth_config,
)
from app.core.config import Settings
from app.core.errors import ValidationError
from app.settings.schemas import (
    AnnotationSettingsResponse,
    AnnotationSettingsUpdate,
    DeploymentInfoResponse,
    DeploymentItem,
    LLMConnectionTestRequest,
    LLMConnectionTestResult,
    LLMSettingsInternal,
    LLMSettingsOut,
    LLMSettingsUpdate,
    PipelineSettingsOut,
    PipelineSettingsUpdate,
    RuntimeSettingsOut,
    RuntimeSettingsUpdate,
    SamSettingsResponse,
    SamSettingsUpdate,
)


class SettingsService:
    """Persist workspace settings in `{data_dir}/settings.json`.

    Agent LLM credentials may live here (never in SQLite). Env vars remain defaults.
    """

    _DEVICE = re.compile(r"^(?:auto|cpu|mps|cuda(?::\d+)?|\d+(?:,\d+)*)$", re.IGNORECASE)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = (settings.data_dir / "settings.json").resolve()
        self.env_model = (settings.sam_model or "").strip()
        self.env_device = settings.sam_device.strip().lower()
        self.env_img_size = settings.sam_img_size

    # --- SAM (existing Starter) ---

    def get_sam(self) -> SamSettingsResponse:
        section = self._read().get("sam", {})
        model = str(section.get("model", self.env_model)).strip()
        device = str(section.get("device", self.env_device or "auto")).strip().lower()
        try:
            img_size = int(section.get("img_size", self.env_img_size))
        except (TypeError, ValueError):
            img_size = self.env_img_size
        return SamSettingsResponse(
            enabled=bool(section.get("enabled", True)),
            model=model,
            device=device,
            img_size=max(64, min(img_size, 4096)),
            fallback_mode="disabled" if section.get("fallback_mode") == "disabled" else "box",
            model_configured=bool(model),
        )

    def update_sam(self, payload: SamSettingsUpdate) -> SamSettingsResponse:
        device = payload.device.strip().lower()
        if not self._DEVICE.fullmatch(device):
            raise ValidationError("invalid_sam_device", "SAM 设备必须是 auto、cpu、mps、cuda[:id] 或 CUDA id 列表。")
        data = self._read()
        data["sam"] = {
            "enabled": payload.enabled,
            "model": payload.model.strip(),
            "device": device,
            "img_size": payload.img_size,
            "fallback_mode": payload.fallback_mode,
        }
        self._write(data)
        from app.sam.service import clear_model_cache

        clear_model_cache()
        return self.get_sam()

    # Compatibility aliases used by existing routers/tests.
    def get(self) -> SamSettingsResponse:
        return self.get_sam()

    def update(self, payload: SamSettingsUpdate) -> SamSettingsResponse:
        return self.update_sam(payload)

    # --- LLM (adapted from upstream SettingsService) ---

    def get_llm_settings(self) -> LLMSettingsOut:
        settings = self.get_llm_settings_internal()
        return LLMSettingsOut(
            enabled=settings.enabled,
            provider=settings.provider,
            api_base=settings.api_base,
            model=settings.model,
            temperature=settings.temperature,
            timeout_seconds=settings.timeout_seconds,
            api_key_configured=bool(settings.api_key),
            auth_scheme=settings.auth_scheme,
            auth_header_name=settings.auth_header_name,
        )

    def get_llm_settings_internal(self) -> LLMSettingsInternal:
        data = self._read()
        llm = data.get("llm", {}) if isinstance(data.get("llm"), dict) else {}
        api_key = self._resolved_api_key(llm)
        raw_header_name = str(
            llm.get("auth_header_name", self.settings.agent_auth_header_name or "")
        ).strip()
        auth_scheme = coalesce_auth_scheme(
            llm.get("auth_scheme") or self.settings.agent_auth_scheme,
            raw_header_name,
        )
        auth_header_name = raw_header_name if auth_scheme == "header" else ""
        enabled_default = self.settings.agent_provider.strip().lower() not in {"", "mock"}
        provider_default = self.settings.agent_provider if enabled_default else "openai-compatible"
        model_default = self.settings.agent_model if enabled_default else "gpt-4o-mini"
        base_default = (self.settings.agent_base_url or "").strip()
        return LLMSettingsInternal(
            enabled=bool(llm["enabled"]) if "enabled" in llm else enabled_default,
            provider=str(llm.get("provider", provider_default) or provider_default),
            api_base=str(llm.get("api_base", base_default) or base_default),
            model=str(llm.get("model", model_default) or model_default),
            temperature=float(llm.get("temperature", self.settings.agent_temperature)),
            timeout_seconds=int(llm.get("timeout_seconds", self.settings.agent_timeout_seconds)),
            api_key=api_key,
            api_key_configured=bool(api_key),
            auth_scheme=auth_scheme,
            auth_header_name=auth_header_name,
        )

    def update_llm_settings(self, payload: LLMSettingsUpdate) -> LLMSettingsOut:
        current = self.get_llm_settings_internal()
        api_key = current.api_key if payload.api_key is None or payload.api_key == "" else payload.api_key
        auth_scheme, auth_header_name = validate_auth_config(payload.auth_scheme, payload.auth_header_name)
        provider = (payload.provider or "openai-compatible").strip() or "openai-compatible"
        data = self._read()
        data["llm"] = {
            "enabled": payload.enabled,
            "provider": provider,
            "api_base": (payload.api_base or "").strip(),
            "api_key": api_key,
            "model": (payload.model or "").strip() or "gpt-4o-mini",
            "temperature": payload.temperature,
            "timeout_seconds": payload.timeout_seconds,
            "auth_scheme": auth_scheme,
            "auth_header_name": auth_header_name,
        }
        self._write(data)
        return self.get_llm_settings()

    def test_llm_connection(self, payload: LLMConnectionTestRequest) -> LLMConnectionTestResult:
        """Probe an endpoint without persisting form values or response bodies."""
        current = self.get_llm_settings_internal()
        api_key = (payload.api_key or "").strip() or current.api_key.strip()
        base = (payload.api_base or "").strip()
        if not base:
            return LLMConnectionTestResult(
                ok=False,
                message="请填写 API Base URL（例如 https://api.openai.com/v1）",
                remote_test_performed=False,
                code="missing_api_base",
            )
        if not api_key and "localhost" not in base and "127.0.0.1" not in base:
            return LLMConnectionTestResult(
                ok=True,
                message="未配置 API Key（可选），已跳过远端连接测试。需要调用 LLM 时再填写密钥即可。",
                remote_test_performed=False,
                code="skipped_no_key",
            )

        model = (payload.model or "").strip() or "gpt-4o-mini"
        url = f"{base.rstrip('/')}/chat/completions"
        auth_scheme, auth_header_name = validate_auth_config(payload.auth_scheme, payload.auth_header_name)
        headers = build_llm_request_headers(api_key, auth_scheme, auth_header_name)
        timeout = httpx.Timeout(payload.timeout_seconds)
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                response = client.post(
                    url,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 8,
                    },
                )
        except httpx.ConnectError:
            return LLMConnectionTestResult(
                ok=False,
                message="无法连接到服务器，请检查 API Base URL。",
                remote_test_performed=True,
                code="connect_error",
            )
        except httpx.TimeoutException:
            return LLMConnectionTestResult(
                ok=False,
                message="请求超时，请检查网络、代理或增大 Request Timeout 后重试",
                remote_test_performed=True,
                code="timeout",
            )
        except httpx.RequestError:
            return LLMConnectionTestResult(
                ok=False,
                message="请求失败，请检查 API Base URL 和网络连接。",
                remote_test_performed=True,
                code="request_error",
            )

        if response.is_success:
            return LLMConnectionTestResult(
                ok=True,
                message="连接成功，Chat Completions 可正常使用。",
                remote_test_performed=True,
                code="ok",
            )
        return LLMConnectionTestResult(
            ok=False,
            message=f"API 返回 HTTP {response.status_code}",
            remote_test_performed=True,
            code="http_error",
            http_status=response.status_code,
        )

    # --- （lg0219）智能分析：云端解释（可选） ---

    def get_pipeline_settings(self) -> PipelineSettingsOut:
        section = self._read().get("pipeline", {})
        return PipelineSettingsOut(
            enabled=bool(section.get("enabled", False)),
            cloud_enabled=bool(section.get("cloud_enabled", False)),
            api_base=str(section.get("api_base", "")).strip(),
            model=str(section.get("model", "")).strip(),
            api_key_configured=bool(str(section.get("api_key", "")).strip()),
            timeout_seconds=int(section.get("timeout_seconds", 60)),
            max_tokens=int(section.get("max_tokens", 256)),
        )

    def update_pipeline_settings(self, payload: PipelineSettingsUpdate) -> PipelineSettingsOut:
        data = self._read()
        section = dict(data.get("pipeline") or {})
        section["enabled"] = payload.enabled
        section["cloud_enabled"] = payload.cloud_enabled
        section["api_base"] = payload.api_base.strip()
        section["model"] = payload.model.strip()
        section["timeout_seconds"] = payload.timeout_seconds
        section["max_tokens"] = payload.max_tokens
        # 留空表示沿用已保存的密钥；非空则覆盖
        if payload.api_key is not None and payload.api_key.strip():
            section["api_key"] = payload.api_key.strip()
        if not payload.cloud_enabled:
            section.pop("api_key", None)
        data["pipeline"] = section
        self._write(data)
        return self.get_pipeline_settings()

    def pipeline_cloud_override(self) -> dict[str, Any] | None:
        """返回传给 vlm-bridge 的单次覆盖配置；未启用或配置不全时返回 None。"""
        section = self._read().get("pipeline", {})
        if not bool(section.get("cloud_enabled", False)):
            return None
        api_base = str(section.get("api_base", "")).strip()
        model = str(section.get("model", "")).strip()
        api_key = str(section.get("api_key", "")).strip()
        if not (api_base and model and api_key):
            return None
        return {
            "base_url": api_base,
            "model": model,
            "api_key": api_key,
            "timeout": float(section.get("timeout_seconds", 60)),
            "max_tokens": int(section.get("max_tokens", 256)),
        }

    # --- （lg0219）运行参数：即时生效 ---

    def get_runtime_settings(self) -> RuntimeSettingsOut:
        section = self._read().get("runtime", {})
        default_upload_mb = max(1, self.settings.max_upload_bytes // (1024 * 1024))
        default_video_mb = max(1, self.settings.max_video_upload_bytes // (1024 * 1024))
        return RuntimeSettingsOut(
            yolo_executable=str(section.get("yolo_executable", "")).strip(),
            max_upload_mb=int(section.get("max_upload_mb", default_upload_mb)),
            max_video_upload_mb=int(section.get("max_video_upload_mb", default_video_mb)),
            max_video_duration_seconds=int(
                section.get("max_video_duration_seconds", self.settings.max_video_duration_seconds)
            ),
            model_source=str(section.get("model_source", "official") or "official").strip(),
            model_source_custom=str(section.get("model_source_custom", "")).strip(),
        )

    def update_runtime_settings(self, payload: RuntimeSettingsUpdate) -> RuntimeSettingsOut:
        data = self._read()
        section = dict(data.get("runtime") or {})
        section["yolo_executable"] = payload.yolo_executable.strip()
        section["max_upload_mb"] = payload.max_upload_mb
        section["max_video_upload_mb"] = payload.max_video_upload_mb
        section["max_video_duration_seconds"] = payload.max_video_duration_seconds
        section["model_source"] = payload.model_source.strip() or "official"
        section["model_source_custom"] = payload.model_source_custom.strip()
        data["runtime"] = section
        self._write(data)
        return self.get_runtime_settings()

    def effective_settings(self) -> Settings:
        """把界面显式配置的运行参数叠加到启动配置上；未配置的项保持原值。"""
        section = self._read().get("runtime") or {}
        changes: dict[str, int] = {}
        if "max_upload_mb" in section:
            changes["max_upload_bytes"] = int(section["max_upload_mb"]) * 1024 * 1024
        if "max_video_upload_mb" in section:
            changes["max_video_upload_bytes"] = int(section["max_video_upload_mb"]) * 1024 * 1024
        if "max_video_duration_seconds" in section:
            changes["max_video_duration_seconds"] = int(section["max_video_duration_seconds"])
        return replace(self.settings, **changes) if changes else self.settings

    # --- （lg0219）部署信息：只读展示 ---

    def deployment_info(self) -> DeploymentInfoResponse:
        items = [
            DeploymentItem(
                key="data_dir",
                label="数据目录",
                value=str(self.settings.data_dir),
                note="数据集、模型、导出文件都在这里。修改需设置环境变量 YWA_DATA_DIR 并重启。",
            ),
            DeploymentItem(
                key="import_root",
                label="导入根目录",
                value=str(self.settings.import_root),
                note="服务端目录扫描只允许在此之下。修改需设置 YWA_IMPORT_ROOT 并重启。",
            ),
            DeploymentItem(
                key="database_url",
                label="数据库地址",
                value=self.settings.database_url,
                note="修改需设置 YWA_DATABASE_URL 并重启。",
            ),
            DeploymentItem(
                key="listen",
                label="监听地址",
                value=f"{self.settings.host}:{self.settings.port}",
                note="修改需设置 YWA_HOST / YWA_PORT 并重启。",
            ),
            DeploymentItem(
                key="cors_origins",
                label="跨域白名单",
                value=", ".join(self.settings.cors_origins),
                note="修改需设置 YWA_CORS_ORIGINS 并重启。",
            ),
            DeploymentItem(
                key="bridge_url",
                label="智能分析服务地址",
                value=os.environ.get("YWA_BRIDGE_URL", "http://127.0.0.1:8002"),
                note="修改需设置 YWA_BRIDGE_URL 并重启。",
            ),
        ]
        return DeploymentInfoResponse(items=items)

    def _resolved_api_key(self, llm: dict[str, Any]) -> str:
        raw = llm.get("api_key")
        if raw is not None and str(raw).strip():
            return str(raw).strip()
        return (self.settings.agent_api_key or "").strip()

    def _read(self) -> dict:
        if not self.path.is_file():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValidationError("settings_read_failed", "本地设置文件无法读取。") from exc
        return value if isinstance(value, dict) else {}

    def _write(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.path)
        except OSError as exc:
            raise ValidationError("settings_write_failed", "本地设置无法保存。") from exc


# Backward-compatible name used across SAM routers/tests.
    # （lg0219）标注线宽偏好：与 Sam 同款读写模式，存 settings.json 的 annotation 段。
    def get_annotation_settings(self) -> AnnotationSettingsResponse:
        raw = self._read()
        return AnnotationSettingsResponse(line_width=float(raw.get("annotation", {}).get("line_width", 1.0)))

    def update_annotation_settings(self, payload: AnnotationSettingsUpdate) -> AnnotationSettingsResponse:
        data = self._read()
        annotation = dict(data.get("annotation") or {})
        annotation["line_width"] = payload.line_width
        data["annotation"] = annotation
        self._write(data)
        return AnnotationSettingsResponse(line_width=payload.line_width)


SamSettingsService = SettingsService
