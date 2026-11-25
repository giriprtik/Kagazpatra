import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

// The backend writes a random API token to data/.api_token (0600). The dev proxy
// injects it server-side, so the browser never holds the secret and other web pages
// can't call the API (they don't know the token and can't read this file).
function apiToken() {
  if (process.env.NEPDOC_API_TOKEN) return process.env.NEPDOC_API_TOKEN;
  const dataDir = process.env.NEPDOC_DATA_DIR || path.resolve(__dirname, "../data");
  try {
    return fs.readFileSync(path.join(dataDir, ".api_token"), "utf8").trim();
  } catch {
    console.warn("\n[nepdoc] No API token yet — start the backend first, then restart `npm run dev`.\n");
    return "";
  }
}

const backend = process.env.NEPDOC_BACKEND || `http://127.0.0.1:${process.env.NEPDOC_PORT || 8765}`;
const uiPort = Number(process.env.NEPDOC_UI_PORT || 5173);

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1", // never expose the dev server on the LAN
    port: uiPort,
    strictPort: true,
    proxy: {
      "/api": {
        target: backend,
        changeOrigin: true, // backend sees Host: 127.0.0.1 (passes its DNS-rebinding guard)
        configure: (proxy) => {
          const token = apiToken();
          proxy.on("proxyReq", (req) => token && req.setHeader("Authorization", `Bearer ${token}`));
        },
      },
    },
  },
});
