# 更新日志

## 未发布

- RF-DETR 训练支持按档位选择（nano / small / base / medium / large），首次训练自动下载对应官方预训练权重；也可继续手填本地 `.pth`。

## lg0219-v0.2.0（2026-11-20）

- 建立独立的 Starter 仓库与 SQLite baseline。
- 增加数据集、类别、图片、bbox/polygon 标注、校验与 YOLO 数据交换基础能力。
- 增加浏览器工作台：图片上传、受限目录扫描、手工标注、校验与 YOLO 导入/导出控制。
- 增加本地 detect/segment YOLO 训练任务：FIFO 队列、进程停止控制、日志、进度、摘要与 best/last checkpoint 下载。
- 增加从训练产物创建的受管模型版本、PT 下载、模型元数据生命周期与去重的 ONNX FP32 导出。
- Community v2 增加与固定 YoloWebAgent 快照兼容的 OBB 与 classify 数据集、标注、YOLO 交换和本地训练闭环。
- 增加 segment 的 SAM 辅助建议：可配置 Ultralytics SAM 本地/命名检查点；未配置时明确标识为 review-only box_stub。
- 修复导入根目录内文件软链接指向根目录外仍会被扫描读取的问题。
- 训练工作区支持 Ultralytics 内置的 RT-DETR 检测模型（`rtdetr-l.pt` / `rtdetr-x.pt`），沿用现有引擎、权重下载与推理链路。
- 训练高级参数新增「确定性训练」开关（默认开启），显式落到训练命令；在显卡或 NPU 上遇到不支持确定性算子的模型时可关闭。
- 新增「RT-DETR（独立版 v2）」训练引擎：内嵌 `lyuwenyu/RT-DETR` 的 `rtdetrv2_pytorch` 源码，按结构档位（r18/r34/r50/r101）训练，首次训练自动下载主干预训练权重（走镜像，失败自动换通道）；导出 YOLO 数据集自动转 COCO；训练产物带结构旁注，可直接在应用内推理。
