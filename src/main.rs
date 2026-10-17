use lance_arxiv::inspect_dataset;
use std::{env, fs, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = env::args()
        .nth(1)
        .unwrap_or_else(|| "/4tb/dataset/arxiv-complete".to_owned());
    let root = Path::new(&root);
    let output = inspect_dataset(root)?;
    fs::write("row_headers.json", serde_json::to_vec_pretty(&output)?)?;
    println!("Wrote row_headers.json ({} shards)", output["shard_count"]);
    Ok(())
}
