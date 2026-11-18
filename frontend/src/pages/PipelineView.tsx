import { useCallback, useEffect, useRef, useState } from "react";
import { api, apiUrl } from "../api/client";
import type { Dataset } from "../types";

interface TemplateInfo {
  id: string;
  name: string;
  stages: string[];
}

interface JobMeta {
  id: string;
  status: string;
  template?: string;
  title?: string;
  total: number;
  done: number;
  failed: number;
  detected: number;
  elapsed_sec: number;
  eta_sec?: number;
  error?: string | null;
}

interface JobItem {
  index: number;
  name: string;
  path: string;
  ok: boolean;
  summary?: { count: number; max_conf: number; mask_area_ratio_sum?: number };
  reasoning?: { text?: string };
  timings?: Record<string, number>;
  error?: string;
}

async function jget<T>(path: string): Promise<T> {
  const response = await fetch(apiUrl(path));
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()) as T;
}

export function PipelineView() {
  const [templates, setTemplates] = useState<TemplateInfo[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [template, setTemplate] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [limit, setLimit] = useState("0");
  const [saveImages, setSaveImages] = useState(true);
  const [job, setJob] = useState<JobMeta>();
  const [items, setItems] = useState<JobItem[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [bridgeState, setBridgeState] = useState("检查中…");
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    void (async () => {
      try {
        const health = await jget<{ status: string; device?: string }>("/api/pipeline/health");
        setBridgeState(health.status + (health.device ? " · " + health.device : ""));
      } catch {
        setBridgeState("不可达");
      }
      try {
        const tpl = await jget<{ templates: TemplateInfo[] }>("/api/pipeline/templates");
        setTemplates(tpl.templates);
        setTemplate(tpl.templates[0] ? tpl.templates[0].id : "");
      } catch (err) {
        setError("模板读取失败：" + String(err));
      }
      try {
        const list = await api.listDatasets();
        setDatasets(list);
        setDatasetId(list[0] ? list[0].id : "");
      } catch (err) {
        setError("数据集读取失败：" + String(err));
      }
    })();
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, []);

  const loadItems = useCallback(async (jobId: string) => {
    const data = await jget<{ items: JobItem[] }>("/api/pipeline/batch/" + jobId + "/items?limit=500");
    setItems(data.items);
  }, []);

  const poll = useCallback(
    async (jobId: string) => {
      const meta = await jget<JobMeta>("/api/pipeline/batch/" + jobId);
      setJob(meta);
      if (meta.status === "done" || meta.status === "error" || meta.status === "cancelled") {
        if (timer.current) window.clearInterval(timer.current);
        await loadItems(jobId);
      }
    },
    [loadItems],
  );

  const start = async () => {
    if (!datasetId) {
      setError("先选一个数据集");
      return;
    }
    setBusy(true);
    setError("");
    setItems([]);
    try {
      const response = await fetch(apiUrl("/api/pipeline/batch"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: datasetId,
          template: template,
          limit: Number(limit) || 0,
          save_images: saveImages,
        }),
      });
      const meta = (await response.json()) as JobMeta & { detail?: string };
      if (!response.ok) throw new Error(meta.detail ?? "创建任务失败");
      setJob(meta);
      if (timer.current) window.clearInterval(timer.current);
      timer.current = window.setInterval(() => void poll(meta.id), 1500);
      await poll(meta.id);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const cancel = async () => {
    if (!job) return;
    await fetch(apiUrl("/api/pipeline/batch/" + job.id + "/cancel"), { method: "POST" });
  };

  const pct = job && job.total > 0 ? Math.round((job.done * 100) / job.total) : 0;
  const running = job ? job.status === "running" || job.status === "queued" : false;

  return (
    <section className="pipeline-view" style={{ padding: "18px 22px", display: "grid", gap: 16 }}>
      <div>
        <span className="eyebrow">APPLICATION</span>
        <h1 style={{ margin: "4px 0 6px" }}>智能分析</h1>
        <p className="muted" style={{ margin: 0 }}>
          用训练好的模型批量跑图：YOLO 判定 → SAM 量尺寸 → 灰区可选 VLM 解释。推理由独立服务执行，不影响标注与训练。
        </p>
      </div>

      {error ? <div className="toast error">{error}</div> : null}

      <div className="panel" style={{ padding: 16, display: "grid", gap: 12 }}>
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          <label style={{ display: "grid", gap: 4 }}>
            <span className="muted">模板</span>
            <select value={template} onChange={(event) => setTemplate(event.target.value)}>
              {templates.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label style={{ display: "grid", gap: 4 }}>
            <span className="muted">数据集</span>
            <select value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>
              {datasets.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}（{item.image_count} 张）
                </option>
              ))}
            </select>
          </label>
          <label style={{ display: "grid", gap: 4 }}>
            <span className="muted">上限张数（0 = 全部）</span>
            <input type="number" min={0} value={limit} onChange={(event) => setLimit(event.target.value)} style={{ width: 130 }} />
          </label>
          <label style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 18 }}>
            <input type="checkbox" checked={saveImages} onChange={(event) => setSaveImages(event.target.checked)} />
            <span className="muted">保存标注图</span>
          </label>
          <button className="button primary" style={{ marginTop: 18 }} disabled={busy || running || !datasetId} onClick={() => void start()}>
            {busy || running ? "运行中…" : "开始分析"}
          </button>
          <span className="muted" style={{ marginTop: 18 }}>推理服务：{bridgeState}</span>
        </div>
        <p className="muted" style={{ margin: 0, fontSize: 12 }}>
          单卡串行执行，一批跑完再跑下一批。图片由服务端直接读取，不经过浏览器上传。
        </p>
      </div>

      {job ? (
        <div className="panel" style={{ padding: 16, display: "grid", gap: 10 }}>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <strong>任务 {job.id}</strong>
            <span className="muted">模板 {job.template}</span>
            <span className="muted">状态 {job.status}</span>
            <div style={{ flex: 1 }} />
            <button className="button" disabled={!running} onClick={() => void cancel()}>取消</button>
            <a className="button" href={apiUrl("/api/pipeline/batch/" + job.id + "/export.csv")} target="_blank" rel="noreferrer">导出 CSV</a>
          </div>
          <div style={{ height: 10, background: "#e8edf4", borderRadius: 6, overflow: "hidden" }}>
            <div style={{ width: pct + "%", height: "100%", background: "#2563eb", transition: "width .3s" }} />
          </div>
          <div className="muted" style={{ fontSize: 12 }}>
            进度 {job.done}/{job.total} · 有检出 {job.detected} · 失败 {job.failed} · 已用 {job.elapsed_sec}s
            {job.eta_sec ? " · 预计剩余 " + Math.round(job.eta_sec) + "s" : ""}
          </div>
        </div>
      ) : null}

      {items.length > 0 ? (
        <div className="panel" style={{ padding: 0, overflow: "auto", maxHeight: 520 }}>
          <table className="panel-table" style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", padding: "8px 10px" }}>#</th>
                <th style={{ textAlign: "left", padding: "8px 10px" }}>图片</th>
                <th style={{ textAlign: "left", padding: "8px 10px" }}>状态</th>
                <th style={{ textAlign: "left", padding: "8px 10px" }}>检出</th>
                <th style={{ textAlign: "left", padding: "8px 10px" }}>最高置信</th>
                <th style={{ textAlign: "left", padding: "8px 10px" }}>面积占比</th>
                <th style={{ textAlign: "left", padding: "8px 10px" }}>结论</th>
                <th style={{ textAlign: "left", padding: "8px 10px" }}>耗时</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => {
                const summary = item.summary;
                let ms = 0;
                Object.entries(item.timings ?? {}).forEach(([key, value]) => {
                  if (key.endsWith("_ms") && typeof value === "number") ms += value;
                });
                return (
                  <tr key={item.index} style={{ borderTop: "1px solid #e8edf4" }}>
                    <td style={{ padding: "6px 10px" }}>{item.index + 1}</td>
                    <td style={{ padding: "6px 10px", maxWidth: 320, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={item.path}>
                      {item.name}
                    </td>
                    <td style={{ padding: "6px 10px", color: item.ok ? "#15803d" : "#b91c1c" }}>{item.ok ? "成功" : "失败"}</td>
                    <td style={{ padding: "6px 10px", color: summary && summary.count ? "#b45309" : undefined }}>{summary ? summary.count : ""}</td>
                    <td style={{ padding: "6px 10px" }}>{summary ? summary.max_conf : ""}</td>
                    <td style={{ padding: "6px 10px" }}>
                      {summary && summary.mask_area_ratio_sum !== undefined ? (summary.mask_area_ratio_sum * 100).toFixed(2) + "%" : ""}
                    </td>
                    <td style={{ padding: "6px 10px", maxWidth: 340, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={item.reasoning ? item.reasoning.text ?? "" : ""}>
                      {item.reasoning ? item.reasoning.text ?? "" : ""}
                    </td>
                    <td style={{ padding: "6px 10px" }}>{ms ? ms + "ms" : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
