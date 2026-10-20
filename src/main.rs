use lance_arxiv::inspect_dataset;
use std::{env, fs, path::Path};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("mcp-server") => return lance_arxiv::mcp_server::run().await,
        Some("ingest-full") => {
            let root = args
                .next()
                .unwrap_or_else(|| "/4tb/dataset/arxiv-complete".to_owned());
            return lance_arxiv::ingest::ingest_full_dataset(Path::new(&root)).await;
        }
        Some("ingest-sample") => {
            let source = args.next().unwrap_or_else(|| {
                "/4tb/dataset/arxiv-complete/paper_text/train-00000-of-00050.parquet".to_owned()
            });
            return lance_arxiv::ingest::ingest_sample(Path::new(&source)).await;
        }
        Some("ingest-sample-metadata") => {
            return lance_arxiv::ingest::ingest_sample_metadata().await;
        }
        Some("inspect") | None => {}
        Some(command) => {
            return Err(format!(
                "Unknown command {command:?}; use mcp-server, ingest-full, ingest-sample, ingest-sample-metadata, or inspect"
            )
            .into());
        }
    }

    let root = args
        .next()
        .unwrap_or_else(|| "/4tb/dataset/arxiv-complete".to_owned());
    let root = Path::new(&root);
    let output = inspect_dataset(root)?;
    fs::write("row_headers.json", serde_json::to_vec_pretty(&output)?)?;
    println!("Wrote row_headers.json ({} shards)", output["shard_count"]);
    Ok(())
}
