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

Open the React search UI at `http://127.0.0.1:8000/`. The MCP endpoint is
`http://127.0.0.1:8000/mcp`; the health check is
`http://127.0.0.1:8000/health`. The UI uses the official MCP TypeScript client
and calls only MCP tools for status, search, and paper retrieval; it does not
call LanceDB or a parallel REST API. The Vite dev server proxies `/mcp` to the
Rust process.

For frontend development with Vite hot reload, start `./backend.sh` in one
terminal, then run in another:

```sh
./frontend.sh
```

Open `http://127.0.0.1:5173/`. The frontend always connects directly to
`http://127.0.0.1:8000/mcp`, regardless of its own origin. The Rust server must
be running because it owns the LanceDB connection and MCP tools. Its CORS
policy permits requests from any origin so local frontend launches can connect
without an origin allowlist.

The process opens the database/tables once at startup and retains the LanceDB
connections for the server lifetime. VS Code connects to the MCP endpoint but
does not start or manage the server process.

Environment variables:

- `ARXIV_LANCE_DB`: local LanceDB directory (default `./data/arxiv_lance`)
- `ARXIV_LANCE_TABLE`: table to query (default `arxiv_full_text`)
- `ARXIV_METADATA_TABLE`: metadata table (default `arxiv_metadata`)
- `MCP_ADDR`: bind address (default `127.0.0.1:8000`)

The default bind address is loopback-only. Choose an explicit secure deployment
and access-control layer before binding to a non-loopback interface.

## Tools

- `search_full_text(query, limit?, offset?)`: full-text search, returning paper IDs and
  short snippets plus available title/author/category/DOI metadata.
  `limit` is capped at 10. The response includes `results` and `metadata`
  (`lookup_time_ms`, `result_count`, `offset`, `next_offset`, and `has_more`).
  Use `offset` with `limit` to paginate through matching results.
- `get_paper_text(paper_id, max_chars?)`: return stored paper text, capped at
  12,000 characters per call by default and at most, along with available
  metadata.
- `dataset_status()`: report the configured table and its row count.

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
