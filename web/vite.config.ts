import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const configPath = fileURLToPath(new URL("../config.json", import.meta.url));
const runtimeConfig = JSON.parse(readFileSync(configPath, "utf8")) as {
  frontend: { host: string; port: number };
};

export default defineConfig({
  plugins: [
    react(),
    {
      name: "runtime-config",
      configureServer(server) {
        server.middlewares.use("/runtime-config.json", (_request, response) => {
          response.setHeader("Content-Type", "application/json; charset=utf-8");
          response.end(JSON.stringify(runtimeConfig));
        });
      },
    },
  ],
  optimizeDeps: {
    exclude: ["latex.js"],
  },
  server: {
    host: runtimeConfig.frontend.host,
    port: runtimeConfig.frontend.port,
    strictPort: true,
  },
});
