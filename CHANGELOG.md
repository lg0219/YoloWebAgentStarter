# 更新日志

## 未发布

- RF-DETR 训练支持按档位选择（nano / small / base / medium / large），首次训练自动下载对应官方预训练权重；也可继续手填本地 `.pth`。
- 「下载权重」按钮对 RF-DETR 与独立版 RT-DETR 同样可用：按引擎分派到各自的权重来源，界面状态一致。
- 修复档位型引擎点击开始会整页刷新回数据集页、训练任务未提交的问题。
- 修复 RF-DETR 权重下载写入 `~/.roboflow` 时在受限账户下报 `WinError 5` 的问题：缓存目录默认改到项目内 `weights/rfdetr/`，显式设置 `RF_HOME` 时仍尊重用户选择。
- 修复 RF-DETR 训练启动即失败（缺少 `pytorch_lightning`）的问题：依赖改为 `rfdetr[train,loggers]`，并记录绕开 pip 依赖解析卡住的装法。
- 修复 RF-DETR 训练参数与新版框架不匹配的问题：`num_classes` 改 `class_names`、`device` 改 `accelerator`/`devices`，评估后端固定 `faster_coco_eval`（vernier 需要 numpy 2，与项目基线冲突）。
- RF-DETR 训练产物归位到应用约定：`weights/best.pt`、`weights/last.pt` 与 `results.csv`，并在同目录写引擎旁注，使推理能按 RF-DETR 加载。
- RT-DETR（独立版 v2）改为以官方**完整 COCO 预训练权重**为起点（每档一个，r18 约 77MB），不再只下骨干、不再从零训练。
- 下载器统一：RF-DETR 与 RT-DETR 共用并行分段下载，RT 的「下载权重」按钮也显示实时百分比。
- 修复训练页「实验历史」被内容横向撑开、卡片列数随内容变化的问题（收缩链 + 页面横向溢出保护 + 按钮定宽）。
- 训练预估改为按引擎与实际情况计算：训练张数扣除验证集与未标注图，每步耗时按引擎实测值（CPU 标定）；显卡上未标定的引擎不再给误导性数字。

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

