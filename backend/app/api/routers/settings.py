from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_settings
from app.core.config import Settings
from app.settings.schemas import (
    AnnotationSettingsResponse,
    AnnotationSettingsUpdate,
    DeploymentInfoResponse,
    LLMConnectionTestRequest,
    LLMConnectionTestResult,
    LLMSettingsOut,
    LLMSettingsUpdate,
    PipelineSettingsOut,
    PipelineSettingsUpdate,
    RuntimeSettingsOut,
    RuntimeSettingsUpdate,
    SamSettingsResponse,
    SamSettingsUpdate,
)
from app.settings.service import SettingsService


router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/sam", response_model=SamSettingsResponse)
def get_sam_settings(settings: Settings = Depends(get_settings)) -> SamSettingsResponse:
    return SettingsService(settings).get_sam()


@router.put("/sam", response_model=SamSettingsResponse)
def update_sam_settings(payload: SamSettingsUpdate, settings: Settings = Depends(get_settings)) -> SamSettingsResponse:
    return SettingsService(settings).update_sam(payload)


# （lg0219）标注画布偏好（线宽）读写：GET 取当前值，PUT 保存。
@router.get("/annotation", response_model=AnnotationSettingsResponse)
def get_annotation_settings(settings: Settings = Depends(get_settings)) -> AnnotationSettingsResponse:
    return SettingsService(settings).get_annotation_settings()


@router.put("/annotation", response_model=AnnotationSettingsResponse)
def update_annotation_settings(payload: AnnotationSettingsUpdate, settings: Settings = Depends(get_settings)) -> AnnotationSettingsResponse:
    return SettingsService(settings).update_annotation_settings(payload)


@router.get("/llm", response_model=LLMSettingsOut)
def get_llm_settings(settings: Settings = Depends(get_settings)) -> LLMSettingsOut:
    return SettingsService(settings).get_llm_settings()


@router.put("/llm", response_model=LLMSettingsOut)
def update_llm_settings(payload: LLMSettingsUpdate, settings: Settings = Depends(get_settings)) -> LLMSettingsOut:
    return SettingsService(settings).update_llm_settings(payload)


@router.post("/llm/test", response_model=LLMConnectionTestResult)
def test_llm_connection(
    payload: LLMConnectionTestRequest,
    settings: Settings = Depends(get_settings),
) -> LLMConnectionTestResult:
    return SettingsService(settings).test_llm_connection(payload)


# （lg0219）智能分析：云端解释配置（可选，只作用于「解释」阶段）
@router.get("/pipeline", response_model=PipelineSettingsOut)
def get_pipeline_settings(settings: Settings = Depends(get_settings)) -> PipelineSettingsOut:
    return SettingsService(settings).get_pipeline_settings()


@router.put("/pipeline", response_model=PipelineSettingsOut)
def update_pipeline_settings(
    payload: PipelineSettingsUpdate,
    settings: Settings = Depends(get_settings),
) -> PipelineSettingsOut:
    return SettingsService(settings).update_pipeline_settings(payload)


# （lg0219）运行参数：保存后即时生效
@router.get("/runtime", response_model=RuntimeSettingsOut)
def get_runtime_settings(settings: Settings = Depends(get_settings)) -> RuntimeSettingsOut:
    return SettingsService(settings).get_runtime_settings()


@router.put("/runtime", response_model=RuntimeSettingsOut)
def update_runtime_settings(
    payload: RuntimeSettingsUpdate,
    settings: Settings = Depends(get_settings),
) -> RuntimeSettingsOut:
    return SettingsService(settings).update_runtime_settings(payload)


# （lg0219）部署信息：只读
@router.get("/deployment", response_model=DeploymentInfoResponse)
def get_deployment_info(settings: Settings = Depends(get_settings)) -> DeploymentInfoResponse:
    return SettingsService(settings).deployment_info()
