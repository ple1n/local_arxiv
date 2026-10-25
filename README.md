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
- `test_paper_chunks(paper_id, query, candidate_limit?)`: search a paper's
  retained test chunk table through the same hybrid retrieval pipeline used by
  the `search_hybrid_chunks` tool.
- `search_hybrid_chunks(chunk_table, query, embedding_model, reranker_model?,
  candidate_limit?)`: run lexical and dense retrieval against a named chunk
  table, fuse the ranked candidates with reciprocal rank fusion, then optionally
  rerank them.
- `embed_chunk_batch(chunk_table, chunk_ids, model)`: embed and persist up to 8
  chunks in the selected model's vector column.
- `finalize_chunk_embeddings(chunk_table, model)`: create the model-specific
  cosine vector index after embedding batches finish.
- `embed_texts(texts)`: embed 1 to 8 document passages with the configured LFM
  model through llama-server and return the model and vector dimensions.
- `dataset_status()`: report the configured table and its row count.

In the paper reader, **Chunk paper** replaces that paper's retained
`tmp_test_chunks_*` table with stable chunk IDs, text, and source byte offsets,
then builds its FTS index. **Embed chunks** writes each batch into a
model-specific fixed-size vector column and displays per-chunk progress. The
selected model's vector index is built after the batches complete. Re-chunking
replaces that test table; changing embedding models adds another vector column
without overwriting the existing model's vectors.

The query button uses LanceDB BM25 full-text search and vector search over that
same table, fuses the two ranked lists with RRF (constant 60), and optionally
reranks the fused candidates. The per-paper table is a small integration path
through the same `HybridPipeline` chunk-write, embedding-write, index, and
search methods intended for a corpus chunk table. Corpus callers can provide
their own table name and call `upsert_paper_text` followed by
`embed_paper_chunks` to process bounded batches through those same chunking,
persistence, model-column, and index methods. The test-only
`replace_paper_text` operation replaces its per-paper table. No full-corpus
chunking or embedding import has been started; the existing paper-text search
remains separate and lexical-only until such an import is requested.

The text and metadata live in separate LanceDB tables with native columns,
linked by `paper_id`. Full-text search is indexed on `text`; the metadata table
has a B-tree index on `paper_id`. Metadata is looked up by ID and added to the
MCP response; it is not concatenated into a stored blob.

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
truncates from the right, then sends text to llama-server's
`/v1/embeddings` endpoint. The effective input limit is the smaller of the
requested `ctx` and `models.embedding.maxTokens` (512 by default). Embeddings
are normalized in Rust. Qwen reranking remains on llama-server's `/rerank`
endpoint.

The GGUF and tokenizer are user-managed; the Rust backend does not download
model files. Their paths and the tokenizer's Hugging Face repo are in
`models.embedding`. If the tokenizer file is missing, startup reports a manual
`hf download` command. For the default configuration, that command is:

```sh
hf download LiquidAI/LFM2.5-Embedding-350M tokenizer.json \
  --local-dir /b/models/LFM2.5-Embedding-350M
```

`embedding_server.py` is a separate optional Python service and is not used by
the Rust embedding or reranking pipeline. Do not start it for the LFM setup.
The `embedding` config entry only remains for that standalone service. The
chunk-test `ctx` argument defaults to 32K, but local LFM tokenization caps each
input at the configured model maximum. `list_models` exposes the configured
LFM embedding model and discovers reranking models from llama-server; set
`MODEL_LIST_URL` only when the `/v1/models` endpoint is hosted elsewhere.
`RERANK_API_KEY` can provide bearer authentication when required.
