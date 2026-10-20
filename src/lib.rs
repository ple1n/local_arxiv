use parquet2::read::read_metadata;
use serde_json::{Value, json};
use std::{fs, fs::File, path::Path};

pub mod mcp_server;
pub mod ingest;

pub fn inspect_parquet(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let metadata = read_metadata(&mut file)?;
    let fields: Vec<Value> = metadata
        .schema_descr
        .fields()
        .iter()
        .map(|field| json!({"name": field.name(), "definition": format!("{field:?}")}))
        .collect();

    Ok(json!({
        "file": path.to_string_lossy(),
        "bytes": fs::metadata(path)?.len(),
        "rows": metadata.row_groups.iter().map(|group| group.num_rows()).sum::<usize>(),
        "row_groups": metadata.row_groups.len(),
        "schema_name": metadata.schema_descr.name(),
        "columns": fields,
        "leaf_columns": metadata.schema_descr.columns().iter().map(|column| {
            json!({
                "path": column.path_in_schema.join("."),
                "physical_type": format!("{:?}", column.descriptor.primitive_type.physical_type),
                "logical_type": column.descriptor.primitive_type.logical_type.as_ref().map(|kind| format!("{kind:?}")),
                "max_definition_level": column.descriptor.max_def_level,
                "max_repetition_level": column.descriptor.max_rep_level
            })
        }).collect::<Vec<_>>()
    }))
}

pub fn inspect_dataset(root: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    for group in ["metadata", "paper_text"] {
        let directory = root.join(group);
        let mut shards: Vec<_> = fs::read_dir(directory)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<_, _>>()?;
        shards.sort();
        for shard in shards
            .into_iter()
            .filter(|path| path.extension().is_some_and(|ext| ext == "parquet"))
        {
            eprintln!("Inspecting {}", shard.display());
            files.push(inspect_parquet(&shard)?);
        }
    }
    Ok(json!({
        "dataset_root": root.to_string_lossy(),
        "shard_count": files.len(),
        "shards": files
    }))
}
