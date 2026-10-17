# Workspace preferences

- Treat this crate as an exploratory Rust scripting pad. Prefer Rust for Parquet inspection and use dependencies already available in the crate before adding new ones.
- Keep reusable logic in `src/lib.rs` or focused modules; keep `src/main.rs` as a thin executable entry point.
- For the ArXiv dataset at `/4tb/dataset/arxiv-complete/`, inspect its structure and metadata before proposing ingestion. Save useful inspection results and execution traces in the workspace; include JSON-serialized Parquet row headers/schema information.
- Keep large-data exploration bounded until a full scan is needed. Avoid copying the source dataset or materializing all text just to inspect schemas and metadata.
- Treat LanceDB importing as a next-step design task: first document the dataset shape and relevant ingestion considerations, and do not start a full import without being asked.
- Use Git as a scratchpad: preserve unrelated work, make coherent local commits after completed changes, and do not push unless asked.
- Build Rust code inside the workspace Nix flake with `nix develop --command cargo build --release`; it supplies `PROTOC` with the protobuf include path required by Lance.
