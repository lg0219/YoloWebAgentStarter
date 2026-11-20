import { useEffect, useMemo, useState, type FormEvent, type ReactNode } from "react";
import { api } from "../api/client";
import type { AppLocale } from "../locale";
import { saveLocale } from "../locale";
import type {
  AnnotationSettings,
  DeploymentInfo,
  LLMAuthScheme,
  LLMSettings,
  PipelineSettings,
  RuntimeSettings,
  SamSettings,
  TrainingDevice,
} from "../types";

export type SettingsTab = "llm" | "sam" | "annotation" | "pipeline" | "runtime" | "language";

type LlmFormState = LLMSettings & { api_key: string };

const LLM_AUTH_SCHEMES: LLMAuthScheme[] = ["bearer", "header", "raw_authorization"];

const copy = {
  zh: {
    eyebrow: "WORKSPACE SETTINGS",
    title: "设置",
    subtitle: "管理此设备上的 Agent LLM、辅助标注与工作台偏好。配置保存在本地 data 目录，不会写入 SQLite。",
    local: "本地",
    llmNav: "Agent LLM",
    llmNavDescription: "Provider、模型与 API Key",
    samNav: "SAM 辅助",
    samNavDescription: "模型、设备与建议方式",
    generalNav: "通用设置",
    generalNavDescription: "界面显示偏好",
    llmTitle: "Agent LLM",
    llmSubtitle: "对齐上游设置页：启用后使用 OpenAI 兼容接口；关闭则使用本地规则规划器。密钥只保存在 settings.json。",
    enabled: "启用 LLM Provider",
    enabledHint: "关闭时 Agent 使用本地 mock / 规则规划，不发起远端调用。",
    provider: "Provider",
    model: "模型",
    apiBase: "API Base URL",
    apiKey: "API Key",
    apiKeyKeep: "已配置密钥：留空则保留原值",
    apiKeyOptional: "可选：本地服务可不填",
    authScheme: "鉴权方式",
    authHeaderName: "自定义 Header 名",
    authHeaderHint: "仅 auth_scheme=header 时需要，例如 X-Api-Key",
    temperature: "Temperature",
    timeout: "请求超时（秒）",
    advanced: "高级",
    plannerTitle: "规划预设",
    plannerSubtitle: "仅调整 temperature，需点保存后生效。",
    cost: "省成本",
    balanced: "均衡",
    quality: "高质量",
    test: "测试连接",
    testing: "测试中…",
    save: "保存 LLM 设置",
    saved: "LLM 设置已保存",
    llmReady: "已启用",
    llmLocal: "本地规则",
    llmFooter: "测试不会落盘。空 API Key 在保存时保留已有密钥。",
    samTitle: "SAM 辅助标注",
    samSubtitle: "配置 segment 标注页使用的交互式 SAM 建议。建议结果仍需人工确认后保存。",
    samEnabled: "启用 SAM 辅助",
    modelSam: "模型权重或模型 ID",
    modelHint: "可填写本地 .pt 路径或 Ultralytics 可识别的模型名，例如 sam_b.pt。",
    device: "推理设备",
    imageSize: "推理尺寸",
    fallback: "未配置模型时",
    box: "使用框形 review-only 建议",
    disabled: "禁用回退",
    saveSam: "保存 SAM 设置",
    savedSam: "SAM 设置已保存",
    ready: "已配置真实模型",
    fallbackReady: "使用框形建议",
    disabledStatus: "已关闭",
    samFooter: "SAM 只对 segment 数据集开放。",
    annotationNav: "标注画布",
    annotationNavDescription: "标注线宽与缩放补偿",
    annotationTitle: "标注画布设置",
    annotationSubtitle: "调整标注框线的粗细。数值以屏幕像素为准，放大图片时会自动补偿，保证线宽不随缩放变粗或变细。",
    lineWidth: "标注线宽",
    lineWidthHint: "推荐 1.5～2.5 px；过细难看清，过粗会遮挡小目标。",
    saveAnnotation: "保存标注设置",
    savedAnnotation: "标注设置已保存",
    annotationFooter: "改变线宽后，新画的框线立即使用新值；已有框线不变。",
    pipelineNav: "智能分析",
    pipelineNavDescription: "云端解释（可选）",
    pipelineTitle: "智能分析 · 云端解释",
    pipelineSubtitle: "可选。启用后「解释」这一步交给云端多模态模型；检测与分割仍在本地运行。",
    pipelineCloudLabel: "启用云端解释",
    pipelineCloudHint: "不勾选则使用本地模型，行为不变。",
    pipelineEnabledLabel: "启用智能分析页",
    pipelineEnabledHint: "需要先部署 vlm-bridge 推理服务。未启用时左侧不显示入口，也不产生请求。",
    pipelineApiBase: "接口地址",
    pipelineApiBaseHint: "例如 https://dashscope.aliyuncs.com/compatible-mode/v1",
    pipelineModel: "模型名称",
    pipelineModelHint: "例如 qwen-vl-max",
    pipelineApiKey: "密钥",
    pipelineApiKeyHint: "留空表示沿用已保存的密钥。",
    pipelineTimeout: "超时（秒）",
    pipelineMaxTokens: "最大输出长度",
    savePipeline: "保存智能分析设置",
    savedPipeline: "智能分析设置已保存",
    runtimeNav: "运行参数",
    runtimeNavDescription: "上传限制与部署信息",
    runtimeTitle: "运行参数",
    runtimeSubtitle: "保存后立即生效，无需重启。",
    yoloExecutable: "YOLO 可执行文件",
    yoloExecutableHint: "留空则自动使用仓库内的 .venv。",
    maxUpload: "单次上传上限（MB）",
    maxVideoUpload: "视频上传上限（MB）",
    maxVideoDuration: "视频时长上限（秒）",
    modelSource: "权重下载源",
    modelSourceHint: "训练首次使用某个预训练权重时会从 GitHub 下载，国内常卡住。镜像只是加速通道，文件仍来自官方仓库；镜像失效时切回官方或改自定义。",
    modelSourceCustom: "自定义前缀",
    modelSourceCustomHint: "填完整前缀并保留结尾斜杠，例如 https://你的镜像/ ；程序会拼成「前缀 + https://github.com/...」。",
    sourceOfficial: "官方 GitHub",
    sourceGhfast: "镜像 ghfast.top（快）",
    sourceGhProxy: "镜像 gh-proxy.com（快）",
    sourceGhproxyNet: "镜像 ghproxy.net（较慢）",
    sourceCustom: "自定义",
    saveRuntime: "保存运行参数",
    savedRuntime: "运行参数已保存",
    deploymentTitle: "部署信息",
    deploymentSubtitle: "以下为启动参数，只读；修改需改环境变量并重启服务。",
    generalTitle: "通用设置",
    generalSubtitle: "界面显示偏好，所有设置只保存在本机浏览器中。",
    displayLanguage: "显示语言",
    brandBarLabel: "显示顶部品牌区",
    brandBarHint: "关闭后隐藏顶部的品牌、当前数据集名和本地模式；返回数据集请用左侧「数据集」入口。",
    chinese: "简体中文",
    english: "English",
    applied: "已立即应用",
    loading: "正在读取设置…",
    error: "设置读取失败",
    availableDevices: "当前设备",
    noDevice: "没有额外 GPU，仍可手动填写设备标识。",
    cuda: "CUDA",
    mps: "Apple GPU (MPS)",
    cpu: "CPU",
    auto: "自动选择",
    show: "显示",
    hide: "隐藏",
    testMissingApiBase: "请填写 API Base URL（例如 https://api.openai.com/v1）",
    testSkippedNoKey: "未配置 API Key（可选），已跳过远端连接测试。需要调用 LLM 时再填写密钥即可。",
    testConnectError: "无法连接到服务器，请检查 API Base URL。",
    testTimeout: "请求超时，请检查网络、代理或增大 Request Timeout 后重试",
    testRequestError: "请求失败，请检查 API Base URL 和网络连接。",
    testOk: "连接成功，Chat Completions 可正常使用。",
    testHttpError: "API 返回 HTTP {status}",
  },
  en: {
    eyebrow: "WORKSPACE SETTINGS",
    title: "Settings",
    subtitle: "Manage Agent LLM, assisted annotation, and workspace preferences. Stored in the local data directory — never in SQLite.",
    local: "Local",
    llmNav: "Agent LLM",
    llmNavDescription: "Provider, model, and API key",
    samNav: "SAM assist",
    samNavDescription: "Model, device, and proposal behavior",
    generalNav: "General",
    generalNavDescription: "Workspace display preferences",
    llmTitle: "Agent LLM",
    llmSubtitle: "Upstream-aligned settings: enable OpenAI-compatible chat; disable for the local rule planner. Keys live in settings.json only.",
    enabled: "Enable LLM provider",
    enabledHint: "When off, Agent uses the local mock / rule planner and does not call a remote LLM.",
    provider: "Provider",
    model: "Model",
    apiBase: "API Base URL",
    apiKey: "API Key",
    apiKeyKeep: "Key configured — leave blank to keep it",
    apiKeyOptional: "Optional for local endpoints",
    authScheme: "Auth scheme",
    authHeaderName: "Custom header name",
    authHeaderHint: "Required when auth_scheme is header, e.g. X-Api-Key",
    temperature: "Temperature",
    timeout: "Request timeout (seconds)",
    advanced: "Advanced",
    plannerTitle: "Planner presets",
    plannerSubtitle: "Adjusts temperature only; click Save to persist.",
    cost: "Cost",
    balanced: "Balanced",
    quality: "Quality",
    test: "Test connection",
    testing: "Testing…",
    save: "Save LLM settings",
    saved: "LLM settings saved",
    llmReady: "Enabled",
    llmLocal: "Local rules",
    llmFooter: "Tests do not persist. An empty API key keeps the existing secret on save.",
    samTitle: "SAM assisted annotation",
    samSubtitle: "Configure interactive SAM proposals for segment annotation. Review every proposal before saving it.",
    samEnabled: "Enable SAM assist",
    modelSam: "Model checkpoint or model id",
    modelHint: "Use a local .pt path or an Ultralytics model name such as sam_b.pt.",
    device: "Inference device",
    imageSize: "Inference size",
    fallback: "When no model is configured",
    box: "Use box-shaped review-only proposals",
    disabled: "Disable fallback",
    saveSam: "Save SAM settings",
    savedSam: "SAM settings saved",
    ready: "Real model configured",
    fallbackReady: "Box proposals",
    disabledStatus: "Disabled",
    samFooter: "SAM assist is available for segment datasets only.",
    annotationNav: "Annotation canvas",
    annotationNavDescription: "Line width and zoom compensation",
    annotationTitle: "Annotation canvas settings",
    annotationSubtitle: "Adjust the thickness of annotation boxes. Values are in screen pixels; zooming in automatically compensates so lines stay the same visual thickness.",
    lineWidth: "Annotation line width",
    lineWidthHint: "1.5–2.5 px works well. Too thin is hard to see; too thick hides small objects.",
    saveAnnotation: "Save annotation settings",
    savedAnnotation: "Annotation settings saved",
    annotationFooter: "New boxes use the new width immediately; existing boxes are unchanged.",
    pipelineNav: "Smart analysis",
    pipelineNavDescription: "Optional cloud reasoning",
    pipelineTitle: "Smart analysis · cloud reasoning",
    pipelineSubtitle: "Optional. When enabled, the reasoning step runs on a cloud multimodal model; detection and segmentation stay local.",
    pipelineCloudLabel: "Enable cloud reasoning",
    pipelineCloudHint: "Leave off to keep using the local model; nothing changes.",
    pipelineEnabledLabel: "Enable smart analysis page",
    pipelineEnabledHint: "Requires the vlm-bridge service. When off, the sidebar entry is hidden and no requests are made.",
    pipelineApiBase: "API base URL",
    pipelineApiBaseHint: "For example https://dashscope.aliyuncs.com/compatible-mode/v1",
    pipelineModel: "Model name",
    pipelineModelHint: "For example qwen-vl-max",
    pipelineApiKey: "API key",
    pipelineApiKeyHint: "Leave empty to keep the saved key.",
    pipelineTimeout: "Timeout (seconds)",
    pipelineMaxTokens: "Max output tokens",
    savePipeline: "Save smart analysis settings",
    savedPipeline: "Smart analysis settings saved",
    runtimeNav: "Runtime",
    runtimeNavDescription: "Upload limits and deployment info",
    runtimeTitle: "Runtime settings",
    runtimeSubtitle: "Applied immediately after saving; no restart needed.",
    yoloExecutable: "YOLO executable",
    yoloExecutableHint: "Leave empty to use the repository .venv automatically.",
    maxUpload: "Single upload limit (MB)",
    maxVideoUpload: "Video upload limit (MB)",
    maxVideoDuration: "Video duration limit (seconds)",
    modelSource: "Weight download source",
    modelSourceHint: "The first training run downloads pretrained weights from GitHub, which is unreliable in some networks. Mirrors are only accelerators; files still come from the official repository. Switch back to official if a mirror fails.",
    modelSourceCustom: "Custom prefix",
    modelSourceCustomHint: "Full prefix with trailing slash, e.g. https://your-mirror/ ; it is joined as prefix + https://github.com/...",
    sourceOfficial: "Official GitHub",
    sourceGhfast: "Mirror ghfast.top (fast)",
    sourceGhProxy: "Mirror gh-proxy.com (fast)",
    sourceGhproxyNet: "Mirror ghproxy.net (slower)",
    sourceCustom: "Custom",
    saveRuntime: "Save runtime settings",
    savedRuntime: "Runtime settings saved",
    deploymentTitle: "Deployment info",
    deploymentSubtitle: "Read-only startup parameters. Change the environment variables and restart to modify them.",
    generalTitle: "General settings",
    generalSubtitle: "Workspace display preferences. Preferences are stored in this browser only.",
    displayLanguage: "Display language",
    brandBarLabel: "Show top brand bar",
    brandBarHint: "When off, the brand, current dataset name, and local-mode badge are hidden. Use the Datasets item in the sidebar to go back.",
    chinese: "简体中文",
    english: "English",
    applied: "Applied immediately",
    loading: "Loading settings…",
    error: "Could not load settings",
    availableDevices: "Available devices",
    noDevice: "No extra GPU was detected; you can still enter a device id manually.",
    cuda: "CUDA",
    mps: "Apple GPU (MPS)",
    cpu: "CPU",
    auto: "Auto",
    show: "Show",
    hide: "Hide",
    testMissingApiBase: "Enter an API Base URL (for example https://api.openai.com/v1).",
    testSkippedNoKey: "No API key configured (optional). Remote probe skipped; add a key before calling the LLM.",
    testConnectError: "Could not reach the server. Check the API Base URL.",
    testTimeout: "Request timed out. Check network/proxy or increase Request Timeout.",
    testRequestError: "Request failed. Check the API Base URL and network.",
    testOk: "Connection succeeded. Chat Completions is usable.",
    testHttpError: "API returned HTTP {status}",
  },
} as const;

type SettingsText = (typeof copy)[AppLocale];

function localizeLlmTestMessage(
  text: SettingsText,
  result: { message: string; code?: string; http_status?: number | null },
): string {
  switch (result.code) {
    case "missing_api_base":
      return text.testMissingApiBase;
    case "skipped_no_key":
      return text.testSkippedNoKey;
    case "connect_error":
      return text.testConnectError;
    case "timeout":
      return text.testTimeout;
    case "request_error":
      return text.testRequestError;
    case "ok":
      return text.testOk;
    case "http_error":
      return text.testHttpError.replace("{status}", String(result.http_status ?? ""));
    default:
      return result.message;
  }
}

function emptyLlmForm(): LlmFormState {
  return {
    enabled: false,
    provider: "openai-compatible",
    api_base: "",
    model: "gpt-4o-mini",
    temperature: 0.2,
    timeout_seconds: 60,
    api_key_configured: false,
    auth_scheme: "bearer",
    auth_header_name: "",
    api_key: "",
  };
}

export function SettingsView({
  tab,
  locale,
  onLocaleChange,
  showBrandBar,
  onBrandBarChange,
  onSamSettingsChange,
  onTabChange,
}: {
  tab: SettingsTab;
  locale: AppLocale;
  onLocaleChange: (locale: AppLocale) => void;
  // 2026-11-06 新增：顶部品牌区显示开关（存本机浏览器）
  showBrandBar: boolean;
  onBrandBarChange: (show: boolean) => void;
  onSamSettingsChange?: () => void;
  onTabChange: (tab: SettingsTab) => void;
}) {
  const text = copy[locale];
  return (
    <main className="settings-layout">
      <header className="settings-page-head">
        <div>
          <span className="eyebrow">{text.eyebrow}</span>
          <h1>{text.title}</h1>
          <p>{text.subtitle}</p>
        </div>
        <span className="settings-page-badge">{text.local}</span>
      </header>
      <div className="settings-section-layout">
        <nav className="settings-section-nav" aria-label={text.title}>
          <SettingsNavigationItem
            active={tab === "llm"}
            icon={<LlmIcon />}
            title={text.llmNav}
            description={text.llmNavDescription}
            badge={text.local}
            onClick={() => onTabChange("llm")}
          />
          <SettingsNavigationItem
            active={tab === "sam"}
            icon={<SamIcon />}
            title={text.samNav}
            description={text.samNavDescription}
            badge={text.local}
            onClick={() => onTabChange("sam")}
          />
          <SettingsNavigationItem
            active={tab === "annotation"}
            icon={<AnnotationIcon />}
            title={text.annotationNav}
            description={text.annotationNavDescription}
            onClick={() => onTabChange("annotation")}
          />
          <SettingsNavigationItem
            active={tab === "pipeline"}
            icon={<LlmIcon />}
            title={text.pipelineNav}
            description={text.pipelineNavDescription}
            onClick={() => onTabChange("pipeline")}
          />
          <SettingsNavigationItem
            active={tab === "runtime"}
            icon={<SamIcon />}
            title={text.runtimeNav}
            description={text.runtimeNavDescription}
            onClick={() => onTabChange("runtime")}
          />
          <SettingsNavigationItem
            active={tab === "language"}
            icon={<LanguageIcon />}
            title={text.generalNav}
            description={text.generalNavDescription}
            onClick={() => onTabChange("language")}
          />
        </nav>
        <div className="settings-section-content">
          {tab === "llm" ? (
            <LlmSettingsPanel locale={locale} text={text} />
          ) : tab === "sam" ? (
            <SamSettingsPanel locale={locale} text={text} onSamSettingsChange={onSamSettingsChange} />
          ) : tab === "annotation" ? (
            <AnnotationSettingsPanel locale={locale} text={text} />
          ) : tab === "pipeline" ? (
            <PipelineSettingsPanel locale={locale} text={text} />
          ) : tab === "runtime" ? (
            <RuntimeSettingsPanel locale={locale} text={text} />
          ) : (
            <GeneralSettingsPanel locale={locale} text={text} onLocaleChange={onLocaleChange} showBrandBar={showBrandBar} onBrandBarChange={onBrandBarChange} />
          )}
        </div>
      </div>
    </main>
  );
}

function SettingsNavigationItem({
  active,
  icon,
  title,
  description,
  badge,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  title: string;
  description: string;
  badge?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`settings-section-nav-item ${active ? "active" : ""}`}
      aria-current={active ? "page" : undefined}
      onClick={onClick}
    >
      <span className="settings-nav-icon">{icon}</span>
      <span className="settings-section-nav-copy">
        <strong>{title}</strong>
        <small>{description}</small>
      </span>
      {badge ? <span className="settings-nav-badge">{badge}</span> : null}
    </button>
  );
}

function LlmSettingsPanel({ text }: { locale: AppLocale; text: SettingsText }) {
  const [form, setForm] = useState<LlmFormState>(emptyLlmForm);
  const [busy, setBusy] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [preset, setPreset] = useState<"cost" | "balanced" | "quality">("balanced");

  useEffect(() => {
    void api
      .getLlmSettings()
      .then((settings) => setForm({ ...settings, api_key: "" }))
      .catch((reason) => setError(reason instanceof Error ? reason.message : text.error))
      .finally(() => setBusy(false));
  }, [text.error]);

  const status = form.enabled ? text.llmReady : text.llmLocal;

  const save = async (event?: FormEvent) => {
    event?.preventDefault();
    setSaving(true);
    setError("");
    setMessage("");
    try {
      const saved = await api.updateLlmSettings({
        enabled: form.enabled,
        provider: form.provider,
        api_base: form.api_base,
        api_key: form.api_key,
        model: form.model,
        temperature: form.temperature,
        timeout_seconds: form.timeout_seconds,
        auth_scheme: form.auth_scheme,
        auth_header_name: form.auth_scheme === "header" ? form.auth_header_name : "",
      });
      setForm({ ...saved, api_key: "" });
      setMessage(text.saved);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : text.error);
    } finally {
      setSaving(false);
    }
  };

  const test = async () => {
    setTesting(true);
    setError("");
    setMessage("");
    try {
      const result = await api.testLlmConnection({
        api_base: form.api_base,
        model: form.model,
        api_key: form.api_key,
        timeout_seconds: form.timeout_seconds,
        auth_scheme: form.auth_scheme,
        auth_header_name: form.auth_scheme === "header" ? form.auth_header_name : "",
      });
      const localized = localizeLlmTestMessage(text, result);
      setMessage(localized);
      if (!result.ok) setError(localized);
      else setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : text.error);
    } finally {
      setTesting(false);
    }
  };

  const applyPreset = (next: "cost" | "balanced" | "quality") => {
    setPreset(next);
    const temperature = next === "cost" ? 0.1 : next === "quality" ? 0.9 : 0.4;
    setForm((current) => ({ ...current, temperature }));
  };

  if (busy) return <section className="panel settings-card"><p className="muted settings-loading">{text.loading}</p></section>;

  return (
    <form className="panel settings-card" onSubmit={(event) => void save(event)}>
      <header className="settings-card-head">
        <div className="settings-card-icon"><LlmIcon /></div>
        <div>
          <h2>{text.llmTitle}</h2>
          <p>{text.llmSubtitle}</p>
        </div>
        <span className={`settings-status ${form.enabled ? "ready" : "disabled"}`}>{status}</span>
      </header>
      {error ? <div className="validation invalid"><span>{error}</span></div> : null}
      {message && !error ? <div className="validation valid"><span>{message}</span></div> : null}

      <label className="settings-toggle">
        <input
          type="checkbox"
          checked={form.enabled}
          onChange={(event) => setForm({ ...form, enabled: event.target.checked })}
        />
        <span>
          <strong>{text.enabled}</strong>
          <small>{text.enabledHint}</small>
        </span>
      </label>

      <div className="settings-form-grid">
        <label>
          <span>{text.provider}</span>
          <select value={form.provider} onChange={(event) => setForm({ ...form, provider: event.target.value })}>
            <option value="openai-compatible">openai-compatible</option>
            <option value="openai">openai</option>
            <option value="local">local</option>
          </select>
        </label>
        <label>
          <span>{text.model}</span>
          <input value={form.model} onChange={(event) => setForm({ ...form, model: event.target.value })} />
        </label>
        <label className="settings-span-2">
          <span>{text.apiBase}</span>
          <input
            value={form.api_base}
            placeholder="https://api.openai.com/v1"
            onChange={(event) => setForm({ ...form, api_base: event.target.value })}
          />
        </label>
        <label className="settings-span-2">
          <span>{text.apiKey}</span>
          <div className="settings-key-line">
            <input
              type={showKey ? "text" : "password"}
              value={form.api_key}
              placeholder={form.api_key_configured ? text.apiKeyKeep : text.apiKeyOptional}
              onChange={(event) => setForm({ ...form, api_key: event.target.value })}
              autoComplete="off"
            />
            <button type="button" className="button" onClick={() => setShowKey((value) => !value)}>
              {showKey ? text.hide : text.show}
            </button>
          </div>
        </label>
        <label>
          <span>{text.authScheme}</span>
          <select
            value={form.auth_scheme}
            onChange={(event) => {
              const auth_scheme = event.target.value as LLMAuthScheme;
              setForm({
                ...form,
                auth_scheme,
                auth_header_name: auth_scheme === "header" ? form.auth_header_name : "",
              });
            }}
          >
            {LLM_AUTH_SCHEMES.map((scheme) => (
              <option key={scheme} value={scheme}>{scheme}</option>
            ))}
          </select>
        </label>
        {form.auth_scheme === "header" ? (
          <label>
            <span>{text.authHeaderName}</span>
            <input
              value={form.auth_header_name}
              placeholder="X-Api-Key"
              onChange={(event) => setForm({ ...form, auth_header_name: event.target.value })}
            />
            <small>{text.authHeaderHint}</small>
          </label>
        ) : (
          <div />
        )}
      </div>

      <div className="settings-advanced-title">{text.advanced}</div>
      <div className="settings-form-grid">
        <label>
          <span>{text.temperature} · {form.temperature.toFixed(1)}</span>
          <input
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={form.temperature}
            onChange={(event) => setForm({ ...form, temperature: Number(event.target.value) })}
          />
        </label>
        <label>
          <span>{text.timeout}</span>
          <input
            type="number"
            min={1}
            max={7200}
            value={form.timeout_seconds}
            onChange={(event) => {
              const n = Number(event.target.value);
              setForm({
                ...form,
                timeout_seconds: Number.isFinite(n) ? Math.min(7200, Math.max(1, Math.round(n))) : form.timeout_seconds,
              });
            }}
          />
        </label>
      </div>

      <section className="settings-planner-card">
        <strong>{text.plannerTitle}</strong>
        <p>{text.plannerSubtitle}</p>
        <div className="settings-preset-row">
          {(["cost", "balanced", "quality"] as const).map((item) => (
            <button
              key={item}
              type="button"
              className={`button ${preset === item ? "primary" : ""}`}
              onClick={() => applyPreset(item)}
            >
              {text[item]}
            </button>
          ))}
        </div>
      </section>

      <footer className="settings-footer">
        <span>{text.llmFooter}</span>
        <div className="settings-footer-actions">
          <button type="button" className="button" disabled={testing || saving} onClick={() => void test()}>
            {testing ? text.testing : text.test}
          </button>
          <button type="submit" className="button primary" disabled={saving || testing}>
            {saving ? "…" : text.save}
          </button>
        </div>
      </footer>
    </form>
  );
}

function SamSettingsPanel({ locale, text, onSamSettingsChange }: { locale: AppLocale; text: SettingsText; onSamSettingsChange?: () => void }) {
  const [settings, setSettings] = useState<SamSettings>();
  const [devices, setDevices] = useState<TrainingDevice[]>([]);
  const [busy, setBusy] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    void Promise.all([api.getSamSettings(), api.listTrainingDevices()])
      .then(([nextSettings, nextDevices]) => {
        setSettings(nextSettings);
        setDevices(nextDevices.items);
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : text.error))
      .finally(() => setBusy(false));
  }, [text.error]);

  const deviceOptions = useMemo(
    () => [
      { value: "auto", label: text.auto },
      { value: "cpu", label: text.cpu },
      ...devices.filter((device) => device.type === "mps").map(() => ({ value: "mps", label: text.mps })),
      ...(devices.some((device) => device.type === "cuda") ? [{ value: "cuda", label: text.cuda }] : []),
      ...devices
        .filter((device) => device.type === "cuda")
        .map((device) => ({ value: String(device.index ?? 0), label: `${text.cuda}:${device.index ?? 0} · ${device.name}` })),
    ],
    [devices, text],
  );

  const save = async () => {
    if (!settings) return;
    setSaving(true);
    setError("");
    setMessage("");
    try {
      setSettings(
        await api.updateSamSettings({
          enabled: settings.enabled,
          model: settings.model,
          device: settings.device,
          img_size: settings.img_size,
          fallback_mode: settings.fallback_mode,
        }),
      );
      onSamSettingsChange?.();
      setMessage(text.savedSam);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : text.error);
    } finally {
      setSaving(false);
    }
  };

  if (busy) return <section className="panel settings-card"><p className="muted settings-loading">{text.loading}</p></section>;
  if (!settings) return <section className="panel settings-card"><div className="validation invalid">{error || text.error}</div></section>;
  const status = settings.enabled ? (settings.model_configured ? text.ready : text.fallbackReady) : text.disabledStatus;
  return (
    <section className="panel settings-card">
      <header className="settings-card-head">
        <div className="settings-card-icon"><SamIcon /></div>
        <div>
          <h2>{text.samTitle}</h2>
          <p>{text.samSubtitle}</p>
        </div>
        <span className={`settings-status ${settings.enabled ? "ready" : "disabled"}`}>{status}</span>
      </header>
      {error ? <div className="validation invalid"><span>{error}</span></div> : null}
      {message ? <div className="validation valid"><span>{message}</span></div> : null}
      <div className="settings-form-grid">
        <label className="settings-toggle">
          <input
            type="checkbox"
            checked={settings.enabled}
            onChange={(event) => setSettings({ ...settings, enabled: event.target.checked })}
          />
          <span>
            <strong>{text.samEnabled}</strong>
            <small>{settings.model_configured ? settings.model : text.modelHint}</small>
          </span>
        </label>
        <label>
          <span>{text.modelSam}</span>
          <input value={settings.model} placeholder="sam_b.pt" onChange={(event) => setSettings({ ...settings, model: event.target.value })} />
          <small>{text.modelHint}</small>
        </label>
        <label>
          <span>{text.device}</span>
          <select value={settings.device} onChange={(event) => setSettings({ ...settings, device: event.target.value })}>
            {!deviceOptions.some((option) => option.value === settings.device) ? (
              <option value={settings.device}>{settings.device}</option>
            ) : null}
            {deviceOptions.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
        <label>
          <span>{text.imageSize}</span>
          <input
            type="number"
            min={64}
            max={4096}
            step={64}
            value={settings.img_size}
            onChange={(event) => setSettings({ ...settings, img_size: Number(event.target.value) })}
          />
        </label>
        <label>
          <span>{text.fallback}</span>
          <select
            value={settings.fallback_mode}
            onChange={(event) => setSettings({ ...settings, fallback_mode: event.target.value as SamSettings["fallback_mode"] })}
          >
            <option value="box">{text.box}</option>
            <option value="disabled">{text.disabled}</option>
          </select>
        </label>
      </div>
      {!devices.some((device) => device.type !== "cpu") ? <p className="settings-hint">{text.noDevice}</p> : null}
      <footer className="settings-footer">
        <span>{text.samFooter}</span>
        <button className="button primary" disabled={saving} onClick={() => void save()}>
          {saving ? "…" : text.saveSam}
        </button>
      </footer>
    </section>
  );
}

function AnnotationSettingsPanel({ locale, text }: { locale: AppLocale; text: SettingsText }) {
  const [settings, setSettings] = useState<AnnotationSettings>();
  const [busy, setBusy] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api.getAnnotationSettings()
      .then(setSettings)
      .catch((reason) => setError(reason instanceof Error ? reason.message : text.error))
      .finally(() => setBusy(false));
  }, []);

  // 保存后立即通知画布刷新（AnnotationCanvas 监听同名事件），下次打开设置页也能读到新值。
  const save = async () => {
    if (!settings) return;
    setSaving(true);
    setError("");
    try {
      setSettings(await api.updateAnnotationSettings({ line_width: settings.line_width }));
      window.dispatchEvent(new Event("annotation-settings-changed"));
      setMessage(text.savedAnnotation);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : text.error);
    } finally {
      setSaving(false);
    }
  };

  if (busy) return <section className="panel settings-card"><p className="muted settings-loading">{text.loading}</p></section>;
  if (!settings) return <section className="panel settings-card"><div className="validation invalid">{error || text.error}</div></section>;
  return (
    <section className="panel settings-card">
      <header className="settings-card-head">
        <div className="settings-card-icon"><AnnotationIcon /></div>
        <div>
          <h2>{text.annotationTitle}</h2>
          <p>{text.annotationSubtitle}</p>
        </div>
        <span className="settings-status ready">{text.savedAnnotation}</span>
      </header>
      {error ? <div className="validation invalid"><span>{error}</span></div> : null}
      {message ? <div className="validation valid"><span>{message}</span></div> : null}
      <div className="settings-form-grid">
        <label className="settings-toggle">
          <span>
            <strong>{text.lineWidth}</strong>
            <small>{text.lineWidthHint}</small>
          </span>
          <input
            type="range"
            min={0.5}
            max={3}
            step={0.1}
            value={settings.line_width}
            style={{ width: "180px" }}
            onChange={(event) => setSettings({ ...settings, line_width: Number(event.target.value) })}
          />
          <span style={{ fontWeight: 600, minWidth: "3ch", textAlign: "right" }}>{settings.line_width.toFixed(1)} px</span>
        </label>
      </div>
      <footer className="settings-footer">
        <span>{text.annotationFooter}</span>
        <button className="button primary" disabled={saving} onClick={() => void save()}>{saving ? "…" : text.saveAnnotation}</button>
      </footer>
    </section>
  );
}

function GeneralSettingsPanel({
  locale,
  text,
  onLocaleChange,
  showBrandBar,
  onBrandBarChange,
}: {
  locale: AppLocale;
  text: SettingsText;
  onLocaleChange: (locale: AppLocale) => void;
  showBrandBar: boolean;
  onBrandBarChange: (show: boolean) => void;
}) {
  const change = (next: AppLocale) => {
    saveLocale(next);
    onLocaleChange(next);
  };
  return (
    <section className="panel settings-card language-card">
      <header className="settings-card-head">
        <div className="settings-card-icon"><LanguageIcon /></div>
        <div>
          <h2>{text.generalTitle}</h2>
          <p>{text.generalSubtitle}</p>
        </div>
        <span className="settings-status ready">{text.applied}</span>
      </header>
      <div className="language-card-body">
        <strong>{text.displayLanguage}</strong>
        <div className="language-segment" role="group" aria-label={text.displayLanguage}>
          <button type="button" className={locale === "zh" ? "active" : ""} onClick={() => change("zh")}>{text.chinese}</button>
          <button type="button" className={locale === "en" ? "active" : ""} onClick={() => change("en")}>{text.english}</button>
        </div>
      </div>
      {/* 2026-11-06 新增：顶部品牌区开关。明示隐藏后哪些信息会消失，避免用户误以为功能丢失。 */}
      <div className="brand-bar-toggle">
        <label className="brand-bar-checkbox" htmlFor="brand-bar-visible">
          <input id="brand-bar-visible" type="checkbox" checked={showBrandBar} onChange={(event) => onBrandBarChange(event.target.checked)} />
          <span>
            <strong>{text.brandBarLabel}</strong>
            <small>{text.brandBarHint}</small>
          </span>
        </label>
      </div>
      <footer className="settings-footer"><span>{text.applied}</span></footer>
    </section>
  );
}

function AnnotationIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <rect x="5" y="5" width="14" height="14" rx="3" stroke="currentColor" strokeWidth="2" />
      <path d="M9 9L15 15M9 15L15 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function LlmIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 7h16v10H4zM8 11h8M8 14h5" />
    </svg>
  );
}

function SamIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 5.5h16v13H4zM8 9l2.5 2.5L16 8m-8 7h8" />
    </svg>
  );
}

function LanguageIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8" />
      <path d="M4 12h16M12 4c2.2 2.1 3.3 4.8 3.3 8S14.2 17.9 12 20c-2.2-2.1-3.3-4.8-3.3-8S9.8 6.1 12 4" />
    </svg>
  );
}

// （lg0219）智能分析：云端解释（可选，只作用于「解释」阶段；不启用时行为不变）
function PipelineSettingsPanel({ text }: { locale: AppLocale; text: SettingsText }) {
  const [settings, setSettings] = useState<PipelineSettings>();
  const [apiKey, setApiKey] = useState("");
  const [busy, setBusy] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api.getPipelineSettings()
      .then(setSettings)
      .catch((reason) => setError(reason instanceof Error ? reason.message : text.error))
      .finally(() => setBusy(false));
  }, []);

  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (!settings) return;
    setSaving(true);
    setError("");
    setMessage("");
    try {
      const next = await api.updatePipelineSettings({
        enabled: settings.enabled,
        cloud_enabled: settings.cloud_enabled,
        api_base: settings.api_base,
        model: settings.model,
        api_key: apiKey.trim() || undefined,
        timeout_seconds: settings.timeout_seconds,
        max_tokens: settings.max_tokens,
      });
      setSettings(next);
      setApiKey("");
      setMessage(text.savedPipeline);
      // 让外壳立即刷新「智能分析」入口的显示状态
      window.dispatchEvent(new Event("pipeline-settings-changed"));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : text.error);
    } finally {
      setSaving(false);
    }
  };

  if (busy) return <section className="panel settings-card"><p className="muted settings-loading">{text.loading}</p></section>;
  if (!settings) return <section className="panel settings-card"><div className="validation invalid">{error || text.error}</div></section>;

  return (
    <form className="panel settings-card" onSubmit={(event) => void save(event)}>
      <header className="settings-card-head">
        <div className="settings-card-icon"><LlmIcon /></div>
        <div>
          <h2>{text.pipelineTitle}</h2>
          <p>{text.pipelineSubtitle}</p>
        </div>
        <span className={`settings-status ${settings.enabled ? "ready" : "disabled"}`}>
          {settings.enabled ? "ON" : "OFF"}
        </span>
      </header>
      {error ? <div className="validation invalid"><span>{error}</span></div> : null}
      {message && !error ? <div className="validation valid"><span>{message}</span></div> : null}

      <label className="settings-toggle">
        <input
          type="checkbox"
          checked={settings.enabled}
          onChange={(event) => setSettings({ ...settings, enabled: event.target.checked })}
        />
        <span>
          <strong>{text.pipelineEnabledLabel}</strong>
          <small>{text.pipelineEnabledHint}</small>
        </span>
      </label>

      <label className="settings-toggle">
        <input
          type="checkbox"
          checked={settings.cloud_enabled}
          onChange={(event) => setSettings({ ...settings, cloud_enabled: event.target.checked })}
        />
        <span>
          <strong>{text.pipelineCloudLabel}</strong>
          <small>{text.pipelineCloudHint}</small>
        </span>
      </label>

      <div className="settings-form-grid">
        <label>
          <span>{text.pipelineApiBase}</span>
          <input
            value={settings.api_base}
            placeholder="https://dashscope.aliyuncs.com/compatible-mode/v1"
            onChange={(event) => setSettings({ ...settings, api_base: event.target.value })}
          />
          <small>{text.pipelineApiBaseHint}</small>
        </label>
        <label>
          <span>{text.pipelineModel}</span>
          <input
            value={settings.model}
            placeholder="qwen-vl-max"
            onChange={(event) => setSettings({ ...settings, model: event.target.value })}
          />
          <small>{text.pipelineModelHint}</small>
        </label>
        <label>
          <span>{text.pipelineApiKey}</span>
          <input
            type="password"
            value={apiKey}
            placeholder={settings.api_key_configured ? "••••••••" : ""}
            onChange={(event) => setApiKey(event.target.value)}
          />
          <small>{text.pipelineApiKeyHint}</small>
        </label>
        <label>
          <span>{text.pipelineTimeout}</span>
          <input
            type="number"
            min={5}
            max={600}
            value={settings.timeout_seconds}
            onChange={(event) => setSettings({ ...settings, timeout_seconds: Number(event.target.value) || 60 })}
          />
        </label>
        <label>
          <span>{text.pipelineMaxTokens}</span>
          <input
            type="number"
            min={32}
            max={4096}
            value={settings.max_tokens}
            onChange={(event) => setSettings({ ...settings, max_tokens: Number(event.target.value) || 256 })}
          />
        </label>
      </div>
      <button type="submit" className="button primary" disabled={saving}>
        {saving ? "…" : text.savePipeline}
      </button>
    </form>
  );
}

// （lg0219）运行参数：保存后即时生效；部署信息只读展示
function RuntimeSettingsPanel({ text }: { locale: AppLocale; text: SettingsText }) {
  const [settings, setSettings] = useState<RuntimeSettings>();
  const [deployment, setDeployment] = useState<DeploymentInfo>();
  const [busy, setBusy] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.getRuntimeSettings(), api.getDeploymentInfo()])
      .then(([runtime, info]) => {
        setSettings(runtime);
        setDeployment(info);
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : text.error))
      .finally(() => setBusy(false));
  }, []);

  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (!settings) return;
    setSaving(true);
    setError("");
    setMessage("");
    try {
      setSettings(await api.updateRuntimeSettings(settings));
      setMessage(text.savedRuntime);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : text.error);
    } finally {
      setSaving(false);
    }
  };

  if (busy) return <section className="panel settings-card"><p className="muted settings-loading">{text.loading}</p></section>;
  if (!settings) return <section className="panel settings-card"><div className="validation invalid">{error || text.error}</div></section>;

  return (
    <>
      <form className="panel settings-card" onSubmit={(event) => void save(event)}>
        <header className="settings-card-head">
          <div className="settings-card-icon"><SamIcon /></div>
          <div>
            <h2>{text.runtimeTitle}</h2>
            <p>{text.runtimeSubtitle}</p>
          </div>
        </header>
        {error ? <div className="validation invalid"><span>{error}</span></div> : null}
        {message && !error ? <div className="validation valid"><span>{message}</span></div> : null}
        <div className="settings-form-grid">
          <label>
            <span>{text.yoloExecutable}</span>
            <input
              value={settings.yolo_executable}
              onChange={(event) => setSettings({ ...settings, yolo_executable: event.target.value })}
            />
            <small>{text.yoloExecutableHint}</small>
          </label>
          <label>
            <span>{text.maxUpload}</span>
            <input
              type="number"
              min={1}
              max={4096}
              value={settings.max_upload_mb}
              onChange={(event) => setSettings({ ...settings, max_upload_mb: Number(event.target.value) || 50 })}
            />
          </label>
          <label>
            <span>{text.maxVideoUpload}</span>
            <input
              type="number"
              min={1}
              max={102400}
              value={settings.max_video_upload_mb}
              onChange={(event) => setSettings({ ...settings, max_video_upload_mb: Number(event.target.value) || 2048 })}
            />
          </label>
          <label>
            <span>{text.maxVideoDuration}</span>
            <input
              type="number"
              min={10}
              max={86400}
              value={settings.max_video_duration_seconds}
              onChange={(event) =>
                setSettings({ ...settings, max_video_duration_seconds: Number(event.target.value) || 14400 })
              }
            />
          </label>
          <label>
            <span>{text.modelSource}</span>
            <select
              value={settings.model_source}
              onChange={(event) => setSettings({ ...settings, model_source: event.target.value })}
            >
              <option value="official">{text.sourceOfficial}</option>
              <option value="ghfast">{text.sourceGhfast}</option>
              <option value="gh-proxy">{text.sourceGhProxy}</option>
              <option value="ghproxy-net">{text.sourceGhproxyNet}</option>
              <option value="custom">{text.sourceCustom}</option>
            </select>
            <small>{text.modelSourceHint}</small>
          </label>
          {settings.model_source === "custom" ? (
            <label>
              <span>{text.modelSourceCustom}</span>
              <input
                value={settings.model_source_custom}
                placeholder="https://your-mirror/"
                onChange={(event) => setSettings({ ...settings, model_source_custom: event.target.value })}
              />
              <small>{text.modelSourceCustomHint}</small>
            </label>
          ) : null}
        </div>
        <button type="submit" className="button primary" disabled={saving}>
          {saving ? "…" : text.saveRuntime}
        </button>
      </form>

      <section className="panel settings-card">
        <header className="settings-card-head">
          <div>
            <h2>{text.deploymentTitle}</h2>
            <p>{text.deploymentSubtitle}</p>
          </div>
        </header>
        <div className="settings-form-grid">
          {(deployment?.items ?? []).map((item) => (
            <label key={item.key}>
              <span>{item.label}</span>
              <input value={item.value} readOnly />
              <small>{item.note}</small>
            </label>
          ))}
        </div>
      </section>
    </>
  );
}
