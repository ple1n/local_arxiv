# Local arXiv LanceDB MCP server

This Rust server exposes full-text queries against the local LanceDB table. It
does not call the live arXiv API. The default full-text table is
`arxiv_full_text` in `./data/arxiv_lance`.

## Run

Start the Python service first:

```sh
uv run --extra indexer --extra embedding python embedding_server.py
```

In another terminal, build the React UI once and start the backend:

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
- `embedding.maxConcurrentTexts`: shared Rust semaphore capacity measured in
  texts, not batches or requests. A batch reserves permits for all its texts
  before dispatch and releases them when its embedding request completes
  (default: `32`, minimum: `32`)
- `pythonEmbeddingModel`: exact model ID for the Python provider; it must match
  the Python service's `EMBEDDING_MODEL`
- `llama.host` and `llama.port`: existing llama-server API address for
  Qwen reranking
- `apiBaseUrl`: complete MCP endpoint URL used by the frontend

The backend serves this runtime config to the built frontend; Vite serves the
same root config during development. Restart the relevant process after editing
the file. Keep the backend on loopback unless you have an explicit secure
deployment and access-control layer.

The database path and table names remain backend environment settings:
`ARXIV_LANCE_DB` (default `./data/arxiv_lance`), `ARXIV_LANCE_TABLE`
(default `arxiv_full_text`), and `ARXIV_METADATA_TABLE` (default
`arxiv_metadata`).

## Built data size

The built `./data` directory occupies 244G:

```text
114G    ./data/arxiv_lance/arxiv_full_text.lance
129G    ./data/arxiv_lance/arxiv_metadata.lance
3.1M    ./data/arxiv_lance/tmp_test_chunks_323331312e3137393536.lance
392K    ./data/arxiv_lance/tmp_test_chunks_303830352e31383836.lance
1.3M    ./data/arxiv_lance/tmp_test_chunks_323330382e3032393631.lance
42M     ./data/arxiv_lance/arxiv_chunks.lance
612M    ./data/arxiv_lance/arxiv_abstracts.lance
244G    ./data/arxiv_lance
244G    ./data/
```

Building abstract embeddings for 3 million abstracts completed on an NVIDIA
GeForce RTX 2080 Ti with 6 GB of VRAM in approximately 12 hours.

## Tools

The MCP server searches only data currently loaded in its LanceDB tables; it
may be a sample rather than the complete arXiv corpus. Use `dataset_status` to
check loaded row counts. Before invoking any tool that writes embeddings or
index state, or may create a missing vector index, ask the user and obtain
explicit consent. Search and inspection requests alone do not imply consent
for database changes.

- `search_full_text(query, limit?, page_size?, page?, request_id?)`: full-text search, returning paper IDs and
  short snippets plus available title/author/category/DOI metadata.
  `limit` is the search scope (up to 1000 candidates). `page_size` controls
  how many cached results are sent per response, and `page` selects the
  zero-based result page. The response includes `results` and `metadata`
  (`lookup_time_ms`, `result_count`, `total_results`, `offset`, `next_offset`,
  and `has_more`). It also includes `search_url`, a page URL that reproduces
  the request. The page
  accepts `tool=search_full_text&query=...&limit=...&page_size=...&page=...` and
  `tool=get_paper_text&paper_id=...&max_chars=...` URL parameters to replay
  those MCP requests. Search responses default to the `compact` profile,
  omitting large fields such as abstracts, licenses, and embedding vectors.
  Use `profile=web` for all result fields, or pass an explicit `fields` list
  such as `["paper_id","snippet","cosine_distance"]` to select exact result
  fields. Pagination metadata and the request ID are always returned.
  Responses include a deterministic BLAKE3 `request_id`
  keyed by query, mode, model, and search scope. The server caches the full
  scoped result set for 30 minutes and slices it for each `request_id`, `page`,
  and `page_size`; changing page size or page does not rerun the database
  search. Changing scope creates a new search request and result set.
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
  optionally be reranked. In paginated embedded-paper search, the offset is
  applied to each underlying lexical/vector query before their results are
  fused.
- `search_embedded_papers(query, embedding_model?, limit?, page_size?, page?,
  semantic_only?, request_id?)`: omit `embedding_model` to use
  `google/embeddinggemma-300m`; set it only to search vectors from another
  configured model. Hybrid lexical+semantic search is the default;
  `semantic_only=true` uses vector-only search. An empty query browses papers
  embedded for that model. `limit` is the database search scope (the UI
  offers 20, 100, and 1000); changing scope starts a separate request rather
  than extending or merging the previous result set. `page_size` controls only
  the number of cached hits sent over the wire (the UI offers 20, 50, and 100).
  Follow-up pages pass the deterministic BLAKE3 `request_id` and select a page
  from that request's cached result set. Semantic-only results are sorted by exact cosine
  distance computed from each returned stored f16 vector and the query vector.
  Each semantic hit includes its f16 vector bit patterns and cosine distance;
  this exact distance is distinct from IVF-PQ's approximate candidate ranking.
- `embed_chunk_batch(chunk_table, chunk_ids, model)`: embed and persist up to 8
  chunks in the selected model's vector column.
- `finalize_chunk_embeddings(chunk_table, model)`: create the model-specific
  cosine vector index after embedding batches finish.
- `embed_texts(texts)`: embed 1 to 8 document passages with the configured
  Python model and return the normalized float32 embedding vectors.
- `inspect_abstract_vectors(paper_ids, query?, embedding_model?)`: return the
  already-stored f16 abstract vectors for 1 to 10 paper IDs as compact f16
  bit patterns. If `query` is provided, embed it once and report direct
  dot-product and cosine-similarity values against each stored vector; paper
  vectors are read from LanceDB and are not recomputed.
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
