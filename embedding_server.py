#!/usr/bin/env python3
"""HTTP embedding service backed by Sentence Transformers."""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

from sentence_transformers import SentenceTransformer

MODEL_ID = os.getenv(
    "EMBEDDING_MODEL",
    "alexliap/Qwen3-Embedding-8B-FP8-DYNAMIC",
)
MAX_BODY_BYTES = 1_048_576
MAX_TEXTS = 32

model: SentenceTransformer
encode_lock = Lock()


class EmbeddingHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        self._send_json(
            HTTPStatus.OK,
            {"status": "ok", "model": MODEL_ID},
        )

    def do_POST(self) -> None:
        if self.path != "/embed":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "Content-Length is required"})
            return

        if content_length < 0 or content_length > MAX_BODY_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Request body is too large"})
            return

        if self.headers.get_content_type() != "application/json":
            self._send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Expected application/json"})
            return

        try:
            payload: Any = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
            return

        texts = payload.get("texts") if isinstance(payload, dict) else None
        if (
            not isinstance(texts, list)
            or not 1 <= len(texts) <= MAX_TEXTS
            or any(not isinstance(text, str) or not text.strip() for text in texts)
        ):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"Provide 1 to {MAX_TEXTS} non-empty strings in 'texts'"},
            )
            return

        try:
            with encode_lock:
                vectors = model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except Exception as exc:
            self.log_error("Embedding inference failed: %s", exc)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Embedding inference failed"})
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "model": MODEL_ID,
                "embeddings": vectors.tolist(),
                "dimensions": int(vectors.shape[1]),
            },
        )

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:
        super().log_message(format_string, *args)


def load_listener_config(config_path: Path) -> tuple[str, int]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Could not read runtime config {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in runtime config {config_path}") from exc

    if not isinstance(config, dict):
        raise ValueError("Runtime config must contain a JSON object")
    listener = config.get("embedding")
    if not isinstance(listener, dict):
        raise ValueError("Runtime config must define an 'embedding' object")

    address = listener.get("host")
    port = listener.get("port")
    if not isinstance(address, str) or not address.strip():
        raise ValueError("embedding.host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("embedding.port must be an integer between 1 and 65535")
    return address, port


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local Sentence Transformers embeddings.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Runtime JSON config (default: ./config.json)",
    )
    args = parser.parse_args()

    address, port = load_listener_config(args.config)
    global model
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    server = ThreadingHTTPServer((address, port), EmbeddingHandler)
    print(f"Serving {MODEL_ID} on http://{address}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
