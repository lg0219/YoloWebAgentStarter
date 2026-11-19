import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  // 部署子路径：默认 /ywa/；部署到根路径时设环境变量 VITE_BASE=/
  const env = loadEnv(mode, ".", "VITE_");
  return {
    plugins: [react()],
    // Konva's package main points at its Node adapter, which requires the
    // optional native canvas package. Browser UI and jsdom tests use this build.
    resolve: { alias: [{ find: /^konva$/, replacement: "konva/lib/index.js" }] },
    base: env.VITE_BASE || "/ywa/",
    server: { host: "127.0.0.1", port: 5173 },
  };
});
