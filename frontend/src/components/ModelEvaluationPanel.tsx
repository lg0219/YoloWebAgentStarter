import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { ModelEvaluationRecord, ModelVersion, SplitName } from "../types";
import { EvaluationDetailArtifacts, EvaluationDetailLogs, EvaluationDetailMetrics, EvaluationDetailSamples, EvaluationDetailSummary } from "./evaluation/EvaluationDetailPanels";
import { groupSamples, summarizeSamples } from "./evaluation/evaluationDetailHelpers";
import { FieldHelp } from "./training/FieldHelp";

interface Props {
  model: ModelVersion;
  evaluations: ModelEvaluationRecord[];
  busy: boolean;
  onEvaluate: (split: SplitName, confidence: number, iou: number) => void;
}

export function ModelEvaluationPanel({ model, evaluations, busy, onEvaluate }: Props) {
  const [split, setSplit] = useState<SplitName>("val");
  // 2026-11-06 新增：评估阈值可调。后端 ModelEvaluationRequest 一直支持 confidence/iou，
  // 但前端此前写死默认值 0.25/0.5，导致用户无法通过调低置信度查看漏检样本。
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(0.5);
  const [selectedId, setSelectedId] = useState("");
  const [logs, setLogs] = useState("");
  const selected = useMemo(() => evaluations.find((item) => item.id === selectedId) ?? evaluations[0], [evaluations, selectedId]);

  useEffect(() => {
    if (!selected) { setLogs(""); return; }
    api.modelEvaluationLogs(model.id, selected.id).then((response) => setLogs(response.logs)).catch(() => setLogs(""));
  }, [model.id, selected?.id, selected?.status]);

  const metrics = selected?.result_json.metrics ?? {};
  const artifacts = selected?.result_json.artifacts ?? {};
  const samples = selected?.result_json.error_samples ?? [];
  const summary = summarizeSamples(samples);
  const groups = groupSamples(samples);

  return (
    <section className="evaluation-panel">
      <div className="evaluation-heading">
        <div><span className="eyebrow">LOCAL EVALUATION</span><h3>按已保存 split 评估</h3><p>后台运行上游同款 YOLO val，并保留指标、图表、日志和可审阅错误样本。</p></div>
        <div className="evaluation-create">
          <div className="evaluation-field">
            <label htmlFor="evaluation-split">split</label>
            <select id="evaluation-split" value={split} onChange={(event) => setSplit(event.target.value as SplitName)}><option value="val">val</option><option value="test">test</option><option value="train">train</option></select>
          </div>
          <div className="evaluation-field">
            <span className="evaluation-field-label"><label htmlFor="evaluation-confidence">置信度阈值</label><FieldHelp text="模型有多大把握才敢把目标报出来。调低＝更愿意报，漏检少、误报多；调高＝更保守，误报少、漏检多。默认 0.25；想看漏掉了什么就调到 0.05 左右。" /></span>
            <input id="evaluation-confidence" type="number" min={0.01} max={0.99} step={0.01} value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} />
          </div>
          <div className="evaluation-field">
            <span className="evaluation-field-label"><label htmlFor="evaluation-iou">IoU 阈值</label><FieldHelp text="判断框画得准不准的重叠标准：预测框和真实框的重叠比例超过它才算命中。调高要求框更贴合，指标更严、分数更低；一般用 0.5。" /></span>
            <input id="evaluation-iou" type="number" min={0} max={0.99} step={0.01} value={iou} onChange={(event) => setIou(Number(event.target.value))} />
          </div>
          <button className="button" disabled={busy} onClick={() => onEvaluate(split, confidence, iou)}>创建评估任务</button>
        </div>
      </div>
      {!evaluations.length ? <div className="evaluation-empty">暂无评估任务。</div> : <div className="evaluation-body">
        <div className="evaluation-history">{evaluations.map((item) => <button key={item.id} className={selected?.id === item.id ? "evaluation-task selected" : "evaluation-task"} onClick={() => setSelectedId(item.id)}><strong>{item.split} · {item.created_at.slice(0, 16).replace("T", " ")}</strong><span className={`status ${item.status}`}>{item.status}</span></button>)}</div>
        {selected && <div className="evaluation-detail">
          <div className="evaluation-status"><strong>任务 {selected.status}</strong><span>conf {selected.confidence} · iou {selected.iou}</span>{selected.error_message && <small>{selected.error_message}</small>}</div>
          <EvaluationDetailMetrics metrics={metrics} />
          <EvaluationDetailSummary summary={summary} />
          <EvaluationDetailArtifacts taskId={selected.id} prCurvePath={artifacts.pr_curve} boxPrCurvePath={artifacts.box_pr_curve} maskPrCurvePath={artifacts.mask_pr_curve} confusionMatrixPath={artifacts.confusion_matrix} artifactUrl={(taskId, name) => api.modelEvaluationArtifactUrl(model.id, taskId, name)} />
          <EvaluationDetailSamples groups={groups} />
          <EvaluationDetailLogs logs={logs} />
        </div>}
      </div>}
    </section>
  );
}
