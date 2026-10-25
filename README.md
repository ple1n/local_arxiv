# Local arXiv LanceDB MCP server

This Rust server exposes full-text queries against the local LanceDB table. It
does not call the live arXiv API. The default full-text table is
`arxiv_full_text` in `./data/arxiv_lance`.

## Run

Build the React UI once, then start the HTTP server:

```sh
npm --prefix web install
npm --prefix web run build
./backend.sh
```

Open the React search UI at `http://127.0.0.1:8250/`. The MCP endpoint is
`http://127.0.0.1:8250/mcp`; the health check is
`http://127.0.0.1:8250/health`. The UI uses the official MCP TypeScript client
and calls only MCP tools for status, search, and paper retrieval; it does not
call LanceDB or a parallel REST API. The Vite dev server proxies `/mcp` to the
Rust process.

For frontend development with Vite hot reload, start `./backend.sh` in one
terminal, then run in another:

```sh
./frontend.sh
```

Open `http://127.0.0.1:5173/`. The frontend connects to the configured MCP URL,
regardless of its own origin. The Rust server must
be running because it owns the LanceDB connection and MCP tools. Its CORS
policy permits requests from any origin so local frontend launches can connect
without an origin allowlist.

The process opens the database/tables once at startup and retains the LanceDB
connections for the server lifetime. VS Code connects to the MCP endpoint but
does not start or manage the server process.

Runtime settings are read from the root [`config.json`](./config.json):

- `backend.host` and `backend.port`: Rust server listen address
- `frontend.host` and `frontend.port`: Vite dev-server listen address
- `embedding.host` and `embedding.port`: optional Python embedding-service
  listen address
- `llama.host` and `llama.port`: existing llama-server API address, used for
  LFM embeddings and Qwen reranking
- `models.embedding`: configured LFM model ID, user-managed GGUF path,
  Hugging Face tokenizer repo/path, and model token limit
- `apiBaseUrl`: complete MCP endpoint URL used by the frontend

The backend serves this runtime config to the built frontend; Vite serves the
same root config during development. Restart the relevant process after editing
the file. Keep the backend on loopback unless you have an explicit secure
deployment and access-control layer.

The database path and table names remain backend environment settings:
`ARXIV_LANCE_DB` (default `./data/arxiv_lance`), `ARXIV_LANCE_TABLE`
(default `arxiv_full_text`), and `ARXIV_METADATA_TABLE` (default
`arxiv_metadata`).

## Tools

- `search_full_text(query, limit?, offset?)`: full-text search, returning paper IDs and
  short snippets plus available title/author/category/DOI metadata.
  `limit` is capped at 10. The response includes `results` and `metadata`
  (`lookup_time_ms`, `result_count`, `offset`, `next_offset`, and `has_more`).
  It also includes `search_url`, a page URL that reproduces the request.
  Use `offset` with `limit` to paginate through matching results. The page
  accepts `tool=search_full_text&query=...&limit=...&offset=...` and
  `tool=get_paper_text&paper_id=...&max_chars=...` URL parameters to replay
  those MCP requests.
- `get_paper_text(paper_id, max_chars?)`: return stored paper text, capped at
  12,000 characters per call by default and at most, along with available
  metadata.
- `start_indexing_job(paper_id?, token_budget?, mode?)`: queue either one paper or all
  papers still pending embedding. `mode` is `abstract` (the default) or
  `full_text`. Full-text mode retains the existing chunking and persistent
  chunk-state behavior. Abstract mode embeds the metadata abstract directly.
  The backend owns the job and streams
  progress at `/indexing-jobs/{job_id}/events`, including papers per second,
  elapsed time, estimated time remaining, current paper, and processing phase.
  The job ID is returned before database preparation or pending-work counting.
  The UI immediately shows an indeterminate progress bar, reports abstract-table
  preparation counts, then switches to paper progress while embedding. Vector
  index creation remains an active phase until it finishes; a full paper count
  does not mean the job has completed. Progress heartbeats arrive every second.
  Connection loss, retries, and per-paper errors are shown in the progress panel,
  with a **Reconnect progress** action that reattaches without starting a new job.
  The web UI can pause and resume a job; pausing waits for the in-flight
  embedding request to finish and blocks subsequent embedding requests.
  Single-paper jobs and full-corpus jobs use the same per-paper processing
  routine; only selection and vector-index finalization differ.
- `test_paper_chunks(paper_id, query, candidate_limit?)`: index or resume the
  paper through the shared corpus pipeline, then test its chunks using the
  hybrid retrieval pipeline.
- `chunk_paper(paper_id, token_budget?)`: preview chunk boundaries without
  writing chunks or starting embedding.
- `search_hybrid_chunks(chunk_table, query, embedding_model, mode?,
  reranker_model?, candidate_limit?)`: run lexical and dense retrieval using
  either the named chunk table (`full_text`) or full-paper BM25 search plus
  abstract vectors (`abstract`, the default). Abstract mode does not require
  full-text chunks or chunk vectors. Abstract-mode
  candidates are fused with full-text lexical results by `paper_id`, then can
  optionally be reranked.
- `embed_chunk_batch(chunk_table, chunk_ids, model)`: embed and persist up to 8
  chunks in the selected model's vector column.
- `finalize_chunk_embeddings(chunk_table, model)`: create the model-specific
  cosine vector index after embedding batches finish.
- `embed_texts(texts)`: embed 1 to 8 document passages with the configured LFM
  model through llama-server and return the model and vector dimensions.
- `dataset_status()`: report the configured table and its row count.

The paper reader's **Chunk paper** action previews chunk boundaries.
The main search UI's **Embedded only** toggle searches the selected abstract
embedding subset; with an empty query it browses embedded papers in pages.
The **Embed** button queues a backend job; the browser subscribes to its progress
stream and does not issue embedding batches itself. The main **Embed** button
queues pending papers according to the selected embedding mode; the reader's
**Embed** button queues the selected paper. Abstract-only mode is selected by
default. Full-text mode indexes
papers not marked chunked and embedded in the main text table. Abstract mode
adds model-specific vector columns directly to `arxiv_metadata`; null vectors
and a model-specific readiness flag identify completed embeddings. The **Index
not ready** action explicitly builds or updates the abstract vector index; a
hybrid search never creates an index or falls back to a table scan. The Web UI's
**Embedding mode** selector controls embedding and hybrid dense search; changing
modes never changes the full-text BM25 search path.

The main text table stores nullable `chunked` and `embedded` flags, model ID,
chunk counts, and the latest indexing error. Boolean bitmap indexes support
filtering pending work, while a B-tree index on `paper_id` supports state
updates. A paper already marked `chunked` bypasses chunk creation/upsert and
resumes embedding only for chunks missing vectors. Progress survives backend
restarts in these per-paper fields; starting a new job resumes unfinished work.
Batch embedding failures are retried per chunk so one bad passage does not mark
its whole batch as failed. Full-text vector indexes are finalized after their
embedding job; abstract embedding and vector-index creation are separate
operations.

The query button uses LanceDB BM25 full-text search and vector search over the
global chunk table in full-text mode, fusing the two ranked lists with RRF
(constant 60). In abstract mode, the same full-text BM25 results are fused by
`paper_id` with dense results from model-specific vectors in `arxiv_metadata`, also using RRF 60. Abstract
mode uses the main paper-text BM25 index, without requiring chunk indexing. Both
paths optionally rerank the fused candidates. The one-paper test and corpus job share
the same durable state transitions, chunk table, embedding adapter, and index
finalization path. Full-corpus indexing is started explicitly from the UI; it
does not begin automatically on server startup.

The text and metadata live in separate LanceDB tables with native columns,
linked by `paper_id`. Full-text search is indexed on `text`; the metadata table
has a B-tree index on `paper_id`. Metadata is looked up by ID and added to the
MCP response; it is not concatenated into a stored blob. The metadata table
stores model-specific abstract-vector columns alongside its existing fields.
Abstract text remains sourced from metadata, so it is not duplicated. A
model-specific nullable readiness flag is set with each vector upsert and has
a bitmap index; pending work and indexed coverage use that flag without
scanning the full metadata table.

The MCP tools only see data already present in the selected LanceDB tables.
Sample-scale and full-corpus ingestion use the same table names and shared
streaming/upsert implementation.

To ingest a bounded sample and its matching metadata:

```sh
nix develop --command cargo run --release -- ingest-sample
```

Sample ingestion adds a 100 MiB input cap. Text is committed in batches of up
to 8 MiB with an atomic progress checkpoint. If interrupted, rerun the same
command to resume.

Full ingestion writes under `./data/arxiv_lance` and can be run with:

```sh
nix develop --command cargo run --release -- ingest-full
```

It processes all text shards and the metadata shard incrementally and prints
row progress. Full phase state is at
`./data/arxiv_lance/full_import.progress.json`; text batch state is at
`./data/arxiv_lance/full_text.progress.json`. Rerun the same command after an
interruption to resume. Batch retries use LanceDB `merge_insert` keyed by
`paper_id`, so replaying a committed batch is idempotent when incoming keys are
unique. Upsert is not a uniqueness constraint; the importer checks final row
counts and unique `paper_id` counts before building indexes.

The workspace `.vscode/mcp.json` registers the local HTTP endpoint with VS Code.
Start the server separately; leaving it running keeps the same database/table
connection alive across chat sessions.

## LFM embeddings and reranking

The Rust backend sends LFM embedding requests to the existing llama-server
configured by `llama.host` and `llama.port`. It adds the model's `query: ` or
`document: ` prompt, tokenizes locally with the configured `tokenizer.json`,
and sends text to llama-server's `/v1/embeddings` endpoint only after verifying
the tokenized input length. Paper chunks are split using this same tokenizer
and include the prompt and special tokens in their measured size. The safe
limit is the smaller of the requested `ctx`, `models.embedding.maxTokens` (512
by default), and the model's 768-token context, minus 32 tokens of headroom.
Over-limit direct embedding inputs are rejected rather than silently truncated.
Pending legacy chunks are checked before embedding; oversized chunks are removed
and regenerated with the tokenizer-safe chunker.
All embedding requests share a single scheduler permit, so a new request waits
until the previous embedding request has completed. Embeddings are normalized
in Rust. Qwen reranking remains on llama-server's `/rerank` endpoint.

The GGUF and tokenizer are user-managed; the Rust backend does not download
model files. Their paths and the tokenizer's Hugging Face repo are in
`models.embedding`. If the tokenizer file is missing, startup reports a manual
`hf download` command. For the default configuration, that command is:

```sh
hf download LiquidAI/LFM2.5-Embedding-350M tokenizer.json \
  --local-dir /b/models/LFM2.5-Embedding-350M
```

The Rust backend uses `embedding_server.py` only for CUDA IVF-PQ index builds;
the browser never calls this service. Start it with the `indexer` extra so the
Python worker can open the existing Lance dataset and push build progress and
completion events to the Rust backend. The Rust backend does not poll the worker:

```sh
uv run --extra indexer python embedding_server.py
```

The worker requires CUDA and fails rather than silently building the index on
the CPU. It derives the required PQ subvector count from the vector dimension.
The optional `/embed` endpoint still uses Sentence Transformers; its
`embedding` extra is needed only when that endpoint is used, and the model is
loaded lazily so index-only runs do not reserve GPU memory for it. The Rust
embedding and reranking pipeline remain unchanged. The chunk-test `ctx`
argument defaults to 32K, but local LFM tokenization caps each input at the
configured model maximum. `list_models` exposes the configured LFM embedding
model and discovers reranking models from llama-server; set `MODEL_LIST_URL`
only when the `/v1/models` endpoint is hosted elsewhere. `RERANK_API_KEY` can
provide bearer authentication when required.

Pausing an indexing job pauses embedding batches. An IVF-PQ build already in
progress is not interruptible; the backend rejects pause requests during that
phase rather than reporting a paused job while GPU work continues.
