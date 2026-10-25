use futures::StreamExt;
use lancedb::{
    arrow::{
        arrow_array::{Array, FixedSizeListArray, Float32Array},
        arrow_schema::DataType,
    },
    connect,
    query::{ExecutableQuery, QueryBase, Select},
};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let database_path =
        env::var("ARXIV_LANCE_DB").unwrap_or_else(|_| "./data/arxiv_lance".to_owned());
    let table_name =
        env::var("ARXIV_METADATA_TABLE").unwrap_or_else(|_| "arxiv_metadata".to_owned());
    let database = connect(&database_path).execute().await?;
    let table = database.open_table(&table_name).execute().await?;
    let schema = table.schema().await?;
    let rows = table.count_rows(None).await?;
    println!("table={table_name} rows={rows}");

    let vector_columns: Vec<_> = schema
        .fields()
        .iter()
        .filter_map(|field| match field.data_type() {
            DataType::FixedSizeList(item, dimensions) if item.data_type() == &DataType::Float32 => {
                Some((field.name().clone(), *dimensions as usize))
            }
            _ => None,
        })
        .collect();
    if vector_columns.is_empty() {
        println!("No float32 fixed-size vector columns found.");
        return Ok(());
    }

    for (column, dimensions) in vector_columns {
        let populated = table
            .count_rows(Some(format!("{column} IS NOT NULL")))
            .await?;
        println!(
            "{column}: dimensions={dimensions}, populated={populated}, null={}",
            rows.saturating_sub(populated)
        );
        if populated == 0 {
            continue;
        }
        let mut samples = table
            .query()
            .only_if(format!("{column} IS NOT NULL"))
            .select(Select::columns(&[&column]))
            .limit(64)
            .execute()
            .await?;
        let mut checked = 0usize;
        let mut invalid = 0usize;
        while let Some(batch) = samples.next().await {
            let batch = batch?;
            let vectors = batch
                .column_by_name(&column)
                .ok_or("selected vector column missing from sample")?
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or("vector column is not a fixed-size list")?;
            for row in 0..vectors.len() {
                if vectors.is_null(row) {
                    continue;
                }
                let values = vectors.value(row);
                let values = values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or("vector values are not float32")?;
                if values.len() != dimensions
                    || values.values().iter().any(|value| !value.is_finite())
                {
                    invalid += 1;
                }
                checked += 1;
            }
        }
        println!("  sample_checked={checked}, invalid_vectors={invalid}");
    }
    Ok(())
}
