#!/usr/bin/env python3
"""HTTP embedding service backed by Sentence Transformers."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import random
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

MODEL_ID = os.getenv(
    "EMBEDDING_MODEL",
    "alexliap/Qwen3-Embedding-8B-FP8-DYNAMIC",
)
MAX_BODY_BYTES = 1_048_576
MAX_TEXTS = 32

model: Any = None
encode_lock = Lock()
gpu_lock = Lock()
index_jobs_lock = Lock()
index_jobs: dict[str, dict[str, Any]] = {}
index_executor = ThreadPoolExecutor(max_workers=1)
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_readiness_scope: tuple[str, str] | None = None
_lance_readiness_scope_installed = False
_readiness_row_counts: tuple[int, int] | None = None
_original_lance_to_tensor: Any = None


def _is_loopback_request(handler: BaseHTTPRequestHandler) -> bool:
    try:
        return ipaddress.ip_address(handler.client_address[0]).is_loopback
    except ValueError:
        return False


def _progress_fields(progress: Any) -> tuple[str, int | None, int | None]:
    stage_value = getattr(progress, "stage", None)
    if stage_value is None:
        stage_value = getattr(progress, "message", None)
    stage = str(stage_value if stage_value is not None else progress)[:200]

    completed = getattr(progress, "completed", None)
    total = getattr(progress, "total", None)
    if (
        isinstance(completed, int)
        and not isinstance(completed, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and total > 0
    ):
        return stage, max(0, completed), total

    fraction = getattr(progress, "progress", None)
    if isinstance(fraction, (int, float)) and not isinstance(fraction, bool):
        value = float(fraction)
        if math.isfinite(value):
            scale = 1000 if 0.0 <= value <= 1.0 else 1
            return stage, max(0, min(scale, round(value * scale))), scale
    return stage, None, None


def _suggest_num_sub_vectors(dimension: int) -> int:
    if dimension <= 0:
        raise ValueError(f"Vector dimension must be positive, got {dimension}")
    if dimension % 16 == 0:
        return dimension // 16
    if dimension % 8 == 0:
        return dimension // 8
    return 1


def _suggest_num_partitions(vector_count: int) -> int:
    if vector_count <= 0:
        raise ValueError(f"Vector count must be positive, got {vector_count}")
    return min(max(1, vector_count // 8192), 4096)


def _writable_fsl_to_tensor(arr: Any, dimension: int) -> Any:
    import torch

    values = arr.values.slice(arr.offset * dimension, len(arr) * dimension)
    vectors = values.to_numpy(zero_copy_only=False).reshape(-1, dimension).copy()
    return torch.from_numpy(vectors)


def _writable_lance_to_tensor(
    batch: Any,
    *,
    hf_converter: Any = None,
    use_blob_api: bool = False,
    **kwargs: Any,
) -> Any:
    import pyarrow as pa
    import torch
    from lance.torch import data as lance_torch_data

    columns = batch.column_names if isinstance(batch, pa.RecordBatch) else list(batch)
    ret = {}
    for column in columns:
        arr = batch[column]
        if isinstance(arr.type, pa.FixedShapeTensorType):
            arr = arr.storage
        if pa.types.is_fixed_size_list(arr.type) and (
            pa.types.is_floating(arr.type.value_type)
            or pa.types.is_integer(arr.type.value_type)
        ):
            tensor = _writable_fsl_to_tensor(arr, arr.type.list_size)
        elif (
            pa.types.is_integer(arr.type)
            or pa.types.is_floating(arr.type)
            or pa.types.is_boolean(arr.type)
        ):
            tensor = torch.from_numpy(arr.to_numpy(zero_copy_only=False).copy())
            if kwargs.get("uint64_as_int64", True) and tensor.dtype == torch.uint64:
                tensor = tensor.to(torch.int64)
        else:
            if _original_lance_to_tensor is None:
                raise TypeError(f"Unsupported Lance tensor type: {arr.type}")
            return _original_lance_to_tensor(
                batch,
                hf_converter=hf_converter,
                use_blob_api=use_blob_api,
                **kwargs,
            )
        ret[column] = tensor

    if len(ret) == 1:
        return next(iter(ret.values()))
    return ret


class _ReadinessFlagSampler:
    def __call__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        columns: Any,
        filter: str,
        batch_readahead: int,
        with_row_id: bool,
        **kwargs: Any,
    ) -> Any:
        yield from dataset.to_batches(
            columns=columns,
            filter=filter,
            batch_size=batch_size,
            batch_readahead=batch_readahead,
            with_row_id=with_row_id,
            use_scalar_index=True,
        )


def _install_lance_readiness_scope(
    vector_column: str, ready_column: str | None
) -> None:
    global _lance_readiness_scope_installed, _original_lance_to_tensor, _readiness_scope
    _readiness_scope = (
        (vector_column, ready_column) if ready_column is not None else None
    )
    if _lance_readiness_scope_installed:
        return

    import lance.sampler as lance_sampler
    from lance.torch import data as lance_torch_data
    import lance.vector as lance_vector

    original_dataset = lance_torch_data.LanceDataset
    if _original_lance_to_tensor is None:
        _original_lance_to_tensor = lance_torch_data._to_tensor
    original_filtered_sample = lance_sampler._filtered_efficient_sample
    original_tqdm = lance_vector.tqdm

    class ReadinessScopedDataset(original_dataset):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            scope = _readiness_scope
            kwargs.setdefault("to_tensor_fn", _writable_lance_to_tensor)
            filter_value = kwargs.get("filter")
            if (
                scope is not None
                and isinstance(filter_value, str)
                and filter_value.casefold()
                == f"{scope[0]} is not null".casefold()
            ):
                kwargs["filter"] = f"{scope[1]} = true"
                if not kwargs.get("samples", 0):
                    kwargs["sampler"] = _ReadinessFlagSampler()
            super().__init__(*args, **kwargs)

    def filtered_sample(
        dataset: Any,
        n: int,
        columns: Any,
        batch_size: int,
        target_takes: int,
        filter: str,
    ) -> Any:
        scope = _readiness_scope
        if scope is None or filter.casefold() != f"{scope[1]} = true".casefold():
            yield from original_filtered_sample(
                dataset, n, columns, batch_size, target_takes, filter
            )
            return

        if n <= 0:
            return

        available = dataset.count_rows(filter=filter)
        sample_count = min(n, available)
        if sample_count == 0:
            return

        window_count = min(64, sample_count)
        for window in range(window_count):
            window_size = sample_count // window_count + (
                1 if window < sample_count % window_count else 0
            )
            stratum_start = window * available // window_count
            stratum_end = (window + 1) * available // window_count
            offset = random.randint(
                stratum_start,
                max(stratum_start, stratum_end - window_size),
            )
            yield from dataset.to_batches(
                columns=columns,
                filter=filter,
                offset=offset,
                limit=window_size,
                batch_size=batch_size,
                use_scalar_index=True,
            )

    def scoped_tqdm(*args: Any, **kwargs: Any) -> Any:
        row_counts = _readiness_row_counts
        if row_counts is not None and kwargs.get("total") == row_counts[0]:
            kwargs["total"] = row_counts[1]
        return original_tqdm(*args, **kwargs)

    lance_torch_data.LanceDataset = ReadinessScopedDataset
    lance_sampler._filtered_efficient_sample = filtered_sample
    lance_vector.tqdm = scoped_tqdm
    lance_torch_data._to_tensor = _writable_lance_to_tensor
    _lance_readiness_scope_installed = True


def _run_index_job(
    job_id: str,
    database_path: str,
    table_name: str,
    column_name: str,
) -> None:
    def update_progress(progress: Any) -> None:
        stage, completed, total = _progress_fields(progress)
        with index_jobs_lock:
            job = index_jobs[job_id]
            job.update(
                {
                    "stage": stage,
                    "completed": completed,
                    "total": total,
                }
            )

    try:
        import lancedb
        import torch
        from lance.torch import data as lance_torch_data

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to build the index on CPU")

        global _original_lance_to_tensor
        if _original_lance_to_tensor is None:
            _original_lance_to_tensor = lance_torch_data._to_tensor
        lance_torch_data._fsl_to_tensor = _writable_fsl_to_tensor
        lance_torch_data._to_tensor = _writable_lance_to_tensor
        database = lancedb.connect(database_path)
        table = database.open_table(table_name)
        vector_type = table.schema.field(column_name).type
        dimension = getattr(vector_type, "list_size", None)
        if not isinstance(dimension, int):
            raise TypeError(
                f"Vector column {column_name!r} must be a fixed-size list, got {vector_type}"
            )
        ready_column = f"{column_name}_ready"
        global _readiness_row_counts
        if table.schema.get_field_index(ready_column) >= 0:
            vector_count = table.count_rows(filter=f"{ready_column} = true")
            _readiness_row_counts = (table.count_rows(), vector_count)
            scope_description = ready_column
        else:
            vector_count = table.count_rows(filter=f"{column_name} IS NOT NULL")
            _readiness_row_counts = None
            scope_description = f"{column_name} IS NOT NULL"
        _install_lance_readiness_scope(
            column_name,
            ready_column if table.schema.get_field_index(ready_column) >= 0 else None,
        )
        num_partitions = _suggest_num_partitions(vector_count)
        dataset = table.to_lance()
        with index_jobs_lock:
            index_jobs[job_id]["state"] = "running"
            index_jobs[job_id]["stage"] = (
                f"GPU IVF-PQ training on {vector_count} scoped vectors "
                f"({num_partitions} partitions; filter: {scope_description})"
            )
        with gpu_lock:
            dataset.create_index(
                column=column_name,
                index_type="IVF_PQ",
                metric="cosine",
                accelerator="cuda",
                num_partitions=num_partitions,
                num_sub_vectors=_suggest_num_sub_vectors(dimension),
                replace=True,
                progress_callback=update_progress,
            )
        with index_jobs_lock:
            index_jobs[job_id].update(
                {
                    "state": "completed",
                    "stage": "Index committed",
                    "completed": 1,
                    "total": 1,
                    "error": None,
                }
            )
    except Exception as exc:
        with index_jobs_lock:
            index_jobs[job_id].update(
                {
                    "state": "failed",
                    "stage": "Index build failed",
                    "error": str(exc),
                }
            )
        print(f"IVF-PQ index job {job_id} failed: {exc}", flush=True)
        traceback.print_exc()


class EmbeddingHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "model": MODEL_ID, "model_loaded": model is not None},
            )
            return

        prefix = "/index-jobs/"
        if not self.path.startswith(prefix):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        if not _is_loopback_request(self):
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Index jobs are loopback-only"})
            return

        job_id = self.path[len(prefix) :]
        if not JOB_ID_PATTERN.fullmatch(job_id):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid job ID"})
            return
        with index_jobs_lock:
            job = index_jobs.get(job_id)
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Index job not found"})
                return
            snapshot = dict(job)
        self._send_json(HTTPStatus.OK, snapshot)

    def do_POST(self) -> None:
        if self.path not in {"/embed", "/index-jobs"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        if self.path == "/index-jobs" and not _is_loopback_request(self):
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Index jobs are loopback-only"})
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

        if self.path == "/index-jobs":
            self._start_index_job(payload)
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
            with encode_lock, gpu_lock:
                global model
                if model is None:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
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

    def _start_index_job(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Expected a JSON object"})
            return
        job_id = payload.get("job_id")
        database_path = payload.get("database_path")
        table_name = payload.get("table_name")
        column_name = payload.get("column_name")
        if (
            not isinstance(job_id, str)
            or not JOB_ID_PATTERN.fullmatch(job_id)
            or not isinstance(database_path, str)
            or not database_path.strip()
            or "://" in database_path
            or not isinstance(table_name, str)
            or not table_name.strip()
            or not isinstance(column_name, str)
            or not column_name.strip()
        ):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Provide a valid job_id, local database_path, table_name, and column_name"},
            )
            return

        with index_jobs_lock:
            if job_id in index_jobs:
                self._send_json(HTTPStatus.CONFLICT, {"error": "Index job ID already exists"})
                return
            if any(job["state"] in {"queued", "running"} for job in index_jobs.values()):
                self._send_json(HTTPStatus.CONFLICT, {"error": "An index build is already running"})
                return
            index_jobs[job_id] = {
                "job_id": job_id,
                "state": "queued",
                "stage": "Queued",
                "completed": 0,
                "total": None,
                "error": None,
            }
            try:
                index_executor.submit(
                    _run_index_job,
                    job_id,
                    database_path,
                    table_name,
                    column_name,
                )
            except RuntimeError as exc:
                del index_jobs[job_id]
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
                return
        self._send_json(HTTPStatus.ACCEPTED, {"job_id": job_id})

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
    server = ThreadingHTTPServer((address, port), EmbeddingHandler)
    print(f"Serving the Python embedding/index service on http://{address}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
