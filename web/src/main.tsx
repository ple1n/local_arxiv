import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./style.css";

async function start() {
  const root = document.getElementById("root");
  if (!root) throw new Error("Missing application root element");

  const response = await fetch("/runtime-config.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load runtime config (${response.status})`);
  }
  const config: unknown = await response.json();
  if (
    typeof config !== "object" ||
    config === null ||
    !("apiBaseUrl" in config) ||
    typeof config.apiBaseUrl !== "string"
  ) {
    throw new Error("Runtime config must include a string apiBaseUrl");
  }
  const apiUrl = new URL(config.apiBaseUrl);
  if (!["http:", "https:"].includes(apiUrl.protocol)) {
    throw new Error("apiBaseUrl must use http or https");
  }

  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <App apiBaseUrl={config.apiBaseUrl} />
    </React.StrictMode>,
  );
}

void start().catch((reason: unknown) => {
  const root = document.getElementById("root");
  if (!root) throw reason;
  root.textContent = `Application startup failed: ${
    reason instanceof Error ? reason.message : String(reason)
  }`;
});
