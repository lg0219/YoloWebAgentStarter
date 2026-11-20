"""模型引擎接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.task_types import TaskType


@dataclass(frozen=True)
class EngineInfo:
    """引擎的对外描述，可用于接口展示与能力判断。"""

    key: str
    display_name: str
    task_types: tuple[TaskType, ...]
    weight_suffixes: tuple[str, ...]

    def supports(self, task_type: TaskType) -> bool:
        return task_type in self.task_types


@dataclass(frozen=True)
class TrainSpec:
    """训练参数：与框架无关的描述，由各引擎自行翻译成 argv。"""

    task_type: TaskType
    model: str
    data: str
    run_dir: str
    epochs: int
    img_size: int
    batch_size: int
    device: str
    workers: int
    seed: int
    optimizer: str | None = None
    lr0: float | None = None
    patience: int | None = None
    resume: bool = False
    # 确定性训练：True 时尽量保证同种子同结果；关掉可提速，也用于个别模型不吃这套算法的情况
    deterministic: bool | None = None


class ModelEngine(ABC):
    """模型引擎适配器。

    接入新框架时实现本接口并注册即可；领域服务只依赖这里的方法。
    """

    info: EngineInfo

    # 该引擎实际生效的训练参数（前端据此显示/隐藏控件）
    supported_params: frozenset[str] = frozenset(
        {
            "model",
            "epochs",
            "batch_size",
            "img_size",
            "workers",
            "seed",
            "patience",
            "optimizer",
            "lr0",
            "device",
            "deterministic",
        }
    )
    # 引擎固定、不可调的参数（前端以只读方式展示）
    fixed_params: dict[str, object] = {}

    @abstractmethod
    def command_prefix(self) -> list[str]:
        """训练 / 评估子进程的命令前缀（argv 前若干项）。"""

    @abstractmethod
    def build_train_argv(self, spec: TrainSpec) -> list[str]:
        """生成训练命令：第 0 项必须是引擎 key，其后为传给引擎的参数。"""

    @abstractmethod
    def load_model(self, model_id: str, model_path: Path) -> Any:
        """加载模型，允许内部缓存。"""

    @abstractmethod
    def infer(
        self,
        model: Any,
        image_path: Path,
        confidence: float,
        iou: float,
        task_type: str,
    ) -> list:
        """对单张图片推理，返回统一结构的检测条目列表。"""

    def supports_task(self, task_type: TaskType) -> bool:
        return self.info.supports(task_type)

    def accepts_weight(self, name: str) -> bool:
        lowered = (name or "").lower()
        return any(lowered.endswith(suffix) for suffix in self.info.weight_suffixes)
