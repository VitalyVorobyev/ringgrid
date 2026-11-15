import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Built once (`bun run build`) and served by tools/review/server.py from dist/. Everything the
// page loads is bundled here (scripts, styles, the IBM Plex files of @vitavision/ui), so the
// server's CSP can forbid every external origin.
//
// `bun run dev` serves the page on 127.0.0.1:5174 and proxies /api to a running review server
// (REVIEW_API, default http://127.0.0.1:8766), for working on the UI with hot reload.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5174,
    strictPort: true,
    proxy: { "/api": { target: process.env["REVIEW_API"] ?? "http://127.0.0.1:8766", changeOrigin: true } },
  },
  build: { outDir: "dist", assetsInlineLimit: 0, sourcemap: false, chunkSizeWarningLimit: 2000 },
  test: { environment: "happy-dom", include: ["src/**/*.test.{ts,tsx}"] },
});
