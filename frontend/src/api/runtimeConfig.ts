export interface RuntimeConfig {
  /** 后端接口入口地址；留空字符串表示与页面同源（需反向代理转发 /api）。 */
  apiBase?: string;
}

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function injectedConfig(): RuntimeConfig {
  const value = (globalThis as { __YWA_CONFIG__?: unknown }).__YWA_CONFIG__;
  return typeof value === "object" && value !== null ? (value as RuntimeConfig) : {};
}

/**
 * 接口入口地址的解析顺序：
 * 1. 页面注入的运行时配置（public/config.js），部署时修改即可，无需重新构建；
 * 2. 构建期变量 VITE_API_BASE；
 * 3. 内置默认值，仅适用于本机开发。
 */
export function resolveApiBase(): string {
  const injected = injectedConfig().apiBase;
  const fromEnv = import.meta.env.VITE_API_BASE as string | undefined;
  const raw = injected ?? fromEnv ?? DEFAULT_API_BASE;
  return raw.trim().replace(/\/$/, "");
}

export function apiUrl(path: string): string {
  return `${resolveApiBase()}${path}`;
}
