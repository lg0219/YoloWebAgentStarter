import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import type { Dataset, TaskType, TrainingEngineCapability, WeightStatus } from "../../types";
import { api } from "../../api/client";
import {
  BATCH_OPTIONS,
  IMG_SIZE_OPTIONS,
  PRESETS,
  defaultModelFor,
  modelChoicesFor,
  taskTypeLabel,
  type PresetId,
  type TrainingFormState,
} from "../../training/helpers";
import { IconBox, IconChevron, IconGrid, IconScan, IconShapes, IconSpark } from "./icons";
import { FieldHelp } from "./FieldHelp";

interface Props {
  datasets: Dataset[];
  dataset: Dataset;
  // （lg0219）可用训练引擎及其能力
  engines: TrainingEngineCapability[];
  form: TrainingFormState;
  preset: PresetId | null;
  advancedOpen: boolean;
  summaryBottomBar: string;
  onDatasetChange: (dataset: Dataset) => void;
  onFormChange: (updater: (prev: TrainingFormState) => TrainingFormState) => void;
  onPreset: (preset: PresetId) => void;
  onAdvancedOpenChange: (open: boolean) => void;
  onSubmit: (event: FormEvent) => void;
}

const PRESET_COPY: Record<PresetId, { title: string; body: string; meta: string }> = {
  fast: { title: "快速验证", body: "适合冒烟与调通链路", meta: "30 ep · batch 16" },
  balanced: { title: "均衡训练", body: "日常默认推荐", meta: "50 ep · batch 16" },
  high: { title: "高精度", body: "更长收敛与更稳指标", meta: "100 ep · batch 8" },
};

const TASK_META: Record<TaskType, { help: string; Icon: typeof IconBox }> = {
  detect: { help: "绝对像素 bbox → YOLO detect 标签", Icon: IconBox },
  segment: { help: "polygon 标注；可选 SAM 建议后确认保存", Icon: IconShapes },
  obb: { help: "中心 / 尺寸 / 角度 → YOLO OBB", Icon: IconScan },
  classify: { help: "每图一类 → YOLO classify 目录布局", Icon: IconGrid },
};

export function TrainingConfigPanel({
  datasets,
  dataset,
  engines,
  form,
  preset,
  advancedOpen,
  summaryBottomBar,
  onDatasetChange,
  onFormChange,
  onPreset,
  onAdvancedOpenChange,
  onSubmit,
}: Props) {
  const setForm = onFormChange;
  const modelChoices = modelChoicesFor(dataset.task_type);
  const modelInList = modelChoices.some((item) => item.value === form.model);
  const taskMeta = TASK_META[dataset.task_type];

  // （lg0219）预训练权重：训练前必须先下载；未就绪时不允许开始训练
  const [weight, setWeight] = useState<WeightStatus | null>(null);
  const [weightBusy, setWeightBusy] = useState(false);
  const [weightError, setWeightError] = useState("");
  const pollTimer = useRef<number | null>(null);

  const stopPolling = () => {
    if (pollTimer.current) {
      window.clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  };

  const refreshWeight = async (model: string) => {
    try {
      const status = await api.getWeightStatus(model);
      setWeight(status);
      if (status.download.status === "downloading") {
        stopPolling();
        pollTimer.current = window.setInterval(async () => {
          try {
            const next = await api.getWeightStatus(model);
            setWeight(next);
            if (next.download.status !== "downloading") stopPolling();
          } catch {
            stopPolling();
          }
        }, 1000);
      } else {
        stopPolling();
      }
    } catch (reason) {
      setWeightError(reason instanceof Error ? reason.message : "权重状态查询失败");
    }
  };

  useEffect(() => {
    setWeight(null);
    setWeightError("");
    void refreshWeight(form.model);
    return stopPolling;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form.model]);

  const downloadWeight = async () => {
    setWeightBusy(true);
    setWeightError("");
    try {
      const state = await api.startWeightDownload(form.model);
      setWeight((prev) => (prev ? { ...prev, download: state } : prev));
      await refreshWeight(form.model);
    } catch (reason) {
      setWeightError(reason instanceof Error ? reason.message : "下载失败");
    } finally {
      setWeightBusy(false);
    }
  };

  const guardSubmit = (event: FormEvent) => {
    if (!weight?.ready) {
      event.preventDefault();
      setWeightError("模型权重未下载，请先下载！");
      return;
    }
    setWeightError("");
    onSubmit(event);
  };
  // 2026-11-05：字段不再整行包 <label>（整行可点、焦点区过大），
  // 改为显式 htmlFor/id 关联，? 帮助图标独立成行内唯一特殊交互点。
  const fieldId = (name: string) => `training-field-${name}-${useId().replace(/:/g, "")}`;
  const ids = {
    experimentName: fieldId("experiment-name"),
    model: fieldId("model"),
    customModel: fieldId("custom-model"),
    epochs: fieldId("epochs"),
    batchSize: fieldId("batch-size"),
    imgSize: fieldId("img-size"),
    valRatio: fieldId("val-ratio"),
    seed: fieldId("seed"),
    lr0: fieldId("lr0"),
    optimizer: fieldId("optimizer"),
    patience: fieldId("patience"),
    workers: fieldId("workers"),
    engine: fieldId("engine"),
    deterministic: fieldId("deterministic"),
  };

  // （lg0219）按所选引擎的能力渲染参数：不支持的隐藏，固定的以只读说明展示
  const engineCap = engines.find((item) => item.key === form.engine);
  const supports = (name: string) => !engineCap || engineCap.supported_params.includes(name);
  const fixedNote = (name: string) => (engineCap?.fixed_params?.[name] as string | undefined) ?? "";
  // 是否使用「官方资产名 + 下载」这套权重形态（YOLO）；RF-DETR 用本地 .pth 路径
  const usesAssets = !engineCap || engineCap.weight_suffixes.includes(".pt");

  const changeEngine = (key: string) => {
    const next = engines.find((item) => item.key === key);
    const nextUsesAssets = !next || next.weight_suffixes.includes(".pt");
    setForm((prev) => ({
      ...prev,
      engine: key,
      // 切换引擎时把权重重置为该引擎可用的默认值，避免残留不匹配的文件名
      model: nextUsesAssets ? defaultModelFor(dataset.task_type) : "",
    }));
  };

  return (
    <section id="training-config-anchor" className="training-ws-card training-ws-config">
      <div className="training-ws-card-head">
        <div>
          <h2>训练配置</h2>
          <p className="training-ws-muted">选择权重与超参后提交，任务进入本机 FIFO 队列由 Ultralytics 执行。</p>
        </div>
      </div>

      <form id="dataset-training-ws-form" className="training-ws-form" onSubmit={guardSubmit}>
        <section className="training-config-block">
          <div className="training-config-block-head">
            <h3>数据集与任务</h3>
            <p>任务类型由数据集锁定；切换数据集会重置表单默认值。</p>
          </div>
          <div className="training-ws-field training-ws-field-wide">
            <span className="training-ws-field-label"><label htmlFor={ids.experimentName}>训练数据集</label></span>
            <select
              id={ids.experimentName}
              value={dataset.id}
              onChange={(event) => {
                const next = datasets.find((item) => item.id === event.target.value);
                if (next) onDatasetChange(next);
              }}
            >
              {datasets.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} · {taskTypeLabel(item.task_type)} · {item.image_count} 张
                </option>
              ))}
            </select>
          </div>
          <div className="tcc-task-type-seg tcc-task-type-seg--wrap" role="group" aria-label="任务类型">
            {(["detect", "segment", "obb", "classify"] as const).map((kind) => {
              const { Icon } = TASK_META[kind];
              return (
                <button
                  key={kind}
                  type="button"
                  className={`tcc-task-type-seg-btn${dataset.task_type === kind ? " active" : ""}`}
                  disabled={dataset.task_type !== kind}
                  aria-pressed={dataset.task_type === kind}
                >
                  <Icon size={15} />
                  <span>{taskTypeLabel(kind)}</span>
                </button>
              );
            })}
          </div>
          <p className="tcc-task-type-detail" role="status">{taskMeta.help}</p>
        </section>

        <section className="training-config-block">
          <div className="training-dataset-ai-card" role="region" aria-label="推荐配置">
            <div className="training-dataset-ai-head">
              <IconSpark size={16} />
              <span className="training-dataset-ai-badge">推荐起点</span>
              <span className="training-dataset-ai-active-pill">本地</span>
            </div>
            <div className="training-recommend-grid">
              <div>
                <small>任务</small>
                <strong>{taskTypeLabel(dataset.task_type)}</strong>
              </div>
              <div>
                <small>权重</small>
                <strong>{form.model || "yolo11n.pt"}</strong>
              </div>
              <div>
                <small>预设</small>
                <strong>均衡训练</strong>
              </div>
              <div>
                <small>数据</small>
                <strong>{dataset.image_count} 图 · {dataset.class_count} 类</strong>
              </div>
            </div>
            <p className="training-dataset-ai-reason">默认复用已保存的 train / val split；可随时覆盖下方参数。</p>
          </div>
        </section>

        <section className="training-config-block">
          <div className="training-config-block-head">
            <h3>训练预设</h3>
          </div>
          <div className="training-dataset-preset-cards" role="group" aria-label="训练预设">
            {(["fast", "balanced", "high"] as const).map((id) => (
              <button
                key={id}
                type="button"
                className={`training-dataset-preset-card${preset === id ? " active" : ""}`}
                onClick={() => onPreset(id)}
              >
                <strong>{PRESET_COPY[id].title}</strong>
                <span className="training-preset-meta">{PRESET_COPY[id].meta}</span>
                <span>{PRESET_COPY[id].body}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="training-config-block">
          <div className="training-config-block-head">
            <h3>模型与基础参数</h3>
          </div>
          <div className="training-ws-form-grid training-ws-form-grid--base">
            <div className="training-ws-field training-ws-field-wide">
              <span className="training-ws-field-label"><label htmlFor={ids.experimentName}>实验名称</label></span>
              <input id={ids.experimentName} value={form.name} onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))} required />
            </div>
            <div className="training-ws-field">
              <span className="training-ws-field-label">
                <label htmlFor={ids.engine}>训练引擎</label>
                <FieldHelp text="用哪个框架训练。YOLO 支持检测/分割/旋转框/分类；RF-DETR 目前只支持检测，且分辨率、优化器、workers 等参数不生效——选中后界面会自动隐藏这些项。" />
              </span>
              <select
                id={ids.engine}
                value={form.engine}
                onChange={(event) => changeEngine(event.target.value)}
              >
                {engines.map((item) => (
                  <option key={item.key} value={item.key}>{item.display_name}</option>
                ))}
                {!engines.some((item) => item.key === form.engine) && (
                  <option value={form.engine}>{form.engine}</option>
                )}
              </select>
            </div>
            {!usesAssets ? (
              <div className="training-ws-field training-ws-field-wide">
                <span className="training-ws-field-label">
                  <label htmlFor={ids.model}>基础权重（.pth）</label>
                  <FieldHelp text="RF-DETR 使用本地 .pth 权重文件。填写完整路径；留空则由框架加载默认预训练权重。" />
                </span>
                <input
                  id={ids.model}
                  value={form.model}
                  placeholder="例如 D:/models/rfdetr-base.pth（留空=框架默认）"
                  onChange={(event) => setForm((prev) => ({ ...prev, model: event.target.value }))}
                />
                <small className={`training-weight-state${weight?.ready ? " ready" : ""}`}>
                  {form.model.trim()
                    ? weight?.ready
                      ? "文件存在，可直接训练"
                      : "文件不存在或尚未检查"
                    : "留空将使用框架默认权重"}
                </small>
              </div>
            ) : (
            <>
            <div className="training-ws-field training-ws-field-wide">
              <span className="training-ws-field-label">
                <label htmlFor={ids.model}>基础权重</label>
                <FieldHelp text="训练的起点模型，相当于站在哪个已有模型的基础上继续练。小模型（n / s）练得快、精度上限低；大模型（m / l / x）更慢、更占显存，但精度上限高。默认 yolo11n.pt；第一次用哪个权重，程序就会自动联网下载哪个。RT-DETR 只能做检测，模型体量大、显存占用高，建议把批大小降到 4 以下。" />
              </span>
              <select
                id={ids.model}
                value={modelInList ? form.model : "__custom__"}
                onChange={(event) => {
                  const value = event.target.value;
                  if (value === "__custom__") return;
                  setForm((prev) => ({ ...prev, model: value }));
                }}
              >
                {modelChoices.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
                {!modelInList && <option value="__custom__">{form.model}</option>}
              </select>
              {/* （lg0219）权重必须先下载；下载中/已就绪时按钮置灰，切换权重后重新判断 */}
              <div className="training-weight-actions">
                <button
                  type="button"
                  className="button"
                  disabled={weightBusy || !weight || weight.ready || weight.download.status === "downloading"}
                  onClick={() => void downloadWeight()}
                >
                  {weight?.download.status === "downloading"
                    ? `下载中 ${weight.download.percent.toFixed(1)}%`
                    : weight?.ready
                      ? "已下载"
                      : weightBusy
                        ? "准备中…"
                        : "下载权重"}
                </button>
                <small className={`training-weight-state${weight?.ready ? " ready" : ""}`}>
                  {weight?.ready
                    ? "本地已就绪，可直接训练"
                    : weight?.download.status === "failed"
                      ? `下载失败：${weight.download.error ?? "未知原因"}`
                      : "尚未下载，请先点击下载"}
                </small>
              </div>
              {weightError ? <small className="annotation-form-error">{weightError}</small> : null}
            </div>
            {!modelInList && (
              <div className="training-ws-field training-ws-field-wide">
                <span className="training-ws-field-label"><label htmlFor={ids.customModel}>自定义权重名</label></span>
                <input id={ids.customModel} value={form.model} onChange={(event) => setForm((prev) => ({ ...prev, model: event.target.value }))} />
              </div>
            )}
            </>
            )}
            <div className="training-ws-field">
              <span className="training-ws-field-label">
                <label htmlFor={ids.epochs}>Epochs</label>
                <FieldHelp text="所有图片完完整整学几轮。学得越久记得越牢，但太久了会把训练图片死记硬背下来（过拟合），到新图片上反而变差。可选 1~10000 轮；预设：快速 30 / 均衡 50 / 高精度 100。" />
              </span>
              <input id={ids.epochs} type="number" min={1} value={form.epochs} onChange={(event) => setForm((prev) => ({ ...prev, epochs: Number(event.target.value) }))} />
            </div>
            <div className="training-ws-field">
              <span className="training-ws-field-label">
                <label htmlFor={ids.batchSize}>Batch size</label>
                <FieldHelp text="一次同时看几张图，然后据此改一次模型。看得多，练得快但更吃显存；看得少，练得慢但更稳。显存不够的时候就把它调小。可选 1~64；预设：快速与均衡 16，高精度 8。" />
              </span>
              <select id={ids.batchSize} value={String(form.batch_size)} onChange={(event) => setForm((prev) => ({ ...prev, batch_size: Number(event.target.value) }))}>
                {!BATCH_OPTIONS.includes(form.batch_size) && <option value={form.batch_size}>{form.batch_size}</option>}
                {BATCH_OPTIONS.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            {supports("img_size") ? (
              <div className="training-ws-field">
                <span className="training-ws-field-label">
                  <label htmlFor={ids.imgSize}>Image size</label>
                  <FieldHelp text="把每张图缩放到多大再喂给模型，单位是边长像素。图越大，小目标越看得清，但更慢、更占显存。一般 640 就够；实在找不到小目标再往上调。可选 256~1024，默认 640。" />
                </span>
                <select id={ids.imgSize} value={String(form.img_size)} onChange={(event) => setForm((prev) => ({ ...prev, img_size: Number(event.target.value) }))}>
                  {!IMG_SIZE_OPTIONS.includes(form.img_size) && <option value={form.img_size}>{form.img_size}px</option>}
                  {IMG_SIZE_OPTIONS.map((n) => <option key={n} value={n}>{n}px</option>)}
                </select>
              </div>
            ) : fixedNote("img_size") ? (
              <div className="training-ws-field">
                <span className="training-ws-field-label"><label>Image size</label></span>
                <small className="training-ws-muted-inline">{fixedNote("img_size")}</small>
              </div>
            ) : null}
          </div>
          <p className="tcc-model-hint training-ws-field-wide" role="note">
            训练前必须先把 {form.model} 下载到本地。点击上方「下载权重」，完成后按钮置灰；切换权重会重新判断是否已下载。请确认适用许可后再训练。
          </p>
        </section>

        <div className="training-ws-advanced">
          <button type="button" className="training-ws-advanced-trigger" onClick={() => onAdvancedOpenChange(!advancedOpen)}>
            <span className={`training-ws-chevron${advancedOpen ? " flip" : ""}`} aria-hidden>
              <IconChevron size={16} />
            </span>
            <span className="training-ws-advanced-trigger-text">
              <strong>高级参数</strong>
              <span className="training-ws-muted-inline">优化器、学习率、patience、workers</span>
            </span>
          </button>
          {advancedOpen && (
            <div className="training-ws-advanced-grid">
              <div className="training-ws-field">
                <span className="training-ws-field-label">
                  <label htmlFor={ids.valRatio}>Val ratio（元数据）</label>
                  <FieldHelp text="它不会真正改变你的训练 / 验证图片分组，只是把这个比例写进这次训练任务的档案里。真正决定用哪些图训练、哪些图验证的，是数据集里每张图已经分好的组；要改分组，去数据集页点「自动切分」。默认 0.2。" />
                </span>
                <input id={ids.valRatio} type="number" step={0.05} min={0} max={0.9} value={form.val_ratio} onChange={(event) => setForm((prev) => ({ ...prev, val_ratio: Number(event.target.value) }))} />
              </div>
              <div className="training-ws-field">
                <span className="training-ws-field-label">
                  <label htmlFor={ids.seed}>Seed</label>
                  <FieldHelp text="训练过程里有很多看似随机的事：图片按什么顺序进场、模型一开始长什么样。这个数字就是这些随机事的剧本编号——填同一个数字，两次训练会走出完全一样的过程，结果可复现、可对比；换个数字，结果就会有细微差别。默认 42。" />
                </span>
                <input id={ids.seed} type="number" value={form.seed} onChange={(event) => setForm((prev) => ({ ...prev, seed: Number(event.target.value) }))} />
              </div>
              {supports("lr0") ? (
                <div className="training-ws-field">
                  <span className="training-ws-field-label">
                    <label htmlFor={ids.lr0}>学习率 lr0</label>
                    <FieldHelp text="模型每改一次自己，改正的幅度。步子太大容易来回震荡、学不稳；太小又学得太慢。一般不用填，让程序自己定；填了必须大于 0。" />
                  </span>
                  <input id={ids.lr0} value={form.lr0} placeholder="留空 = 默认" onChange={(event) => setForm((prev) => ({ ...prev, lr0: event.target.value }))} />
                </div>
              ) : null}
              {supports("optimizer") ? (
                <div className="training-ws-field">
                  <span className="training-ws-field-label">
                    <label htmlFor={ids.optimizer}>Optimizer</label>
                    <FieldHelp text="模型调整自己的方法。默认 auto，让程序按任务自己挑，不用管；要手动指定，就填 SGD 或 AdamW 这类程序认得的名字。" />
                  </span>
                  <input id={ids.optimizer} value={form.optimizer} placeholder="auto / AdamW / SGD" onChange={(event) => setForm((prev) => ({ ...prev, optimizer: event.target.value }))} />
                </div>
              ) : null}
              {supports("patience") ? (
                <div className="training-ws-field">
                  <span className="training-ws-field-label">
                    <label htmlFor={ids.patience}>Patience</label>
                    <FieldHelp text="如果验证效果连续这么多轮都没有进步，就提前收工不练了——既不浪费时间，也防止把训练图片背死。留空按预设：快速 12 / 均衡 20 / 高精度 30。" />
                  </span>
                  <input id={ids.patience} value={form.patience} placeholder={String(PRESETS.balanced.patience)} onChange={(event) => setForm((prev) => ({ ...prev, patience: event.target.value }))} />
                </div>
              ) : null}
              {supports("workers") ? (
                <div className="training-ws-field">
                  <span className="training-ws-field-label">
                    <label htmlFor={ids.workers}>Workers</label>
                    <FieldHelp text="训练时同时运行的进程数——这些进程在后台准备数据（读图、改尺寸、打包好喂给模型）。多开几个，模型就不用干等数据；但开太多会和其他程序抢 CPU。默认 2。" />
                  </span>
                  <input id={ids.workers} type="number" min={0} value={form.workers} onChange={(event) => setForm((prev) => ({ ...prev, workers: Number(event.target.value) }))} />
                </div>
              ) : null}
              {supports("deterministic") ? (
                <div className="training-ws-switch training-ws-field-wide">
                  <input
                    id={ids.deterministic}
                    type="checkbox"
                    checked={form.deterministic}
                    onChange={(event) => setForm((prev) => ({ ...prev, deterministic: event.target.checked }))}
                  />
                  <span className="training-ws-field-label">
                    <label htmlFor={ids.deterministic}>确定性训练</label>
                    <FieldHelp text="开启后，同样的数据配上同样的种子，每次训练都会走出完全一样的过程，方便对比、便于复现。代价是略慢一点，因为程序不再挑那些更快但结果不固定的算法。将来换到显卡或 NPU 上训练 RT-DETR 这类模型时，个别算子不支持确定性计算，建议关掉；关掉不影响训练效果，只是每次结果会有细微差别。Ultralytics 官方默认也是开启，本项目保持一致。" />
                  </span>
                  <small>{form.deterministic ? "已开启：结果可复现，速度略慢" : "已关闭：速度略快，结果不逐次一致"}</small>
                </div>
              ) : null}
            </div>
          )}
        </div>

        <div className="training-ws-meta-row">
          <span>{dataset.name}</span>
          <span>{dataset.image_count} 张 · {dataset.class_count} 类</span>
          <span>{taskTypeLabel(dataset.task_type)}</span>
        </div>

        <div className="training-config-summary-bar">
          <span>{summaryBottomBar}</span>
        </div>
      </form>
    </section>
  );
}
