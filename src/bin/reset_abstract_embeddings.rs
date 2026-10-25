use lancedb::{Table, connect};
use std::env;

const DEFAULT_DATABASE_PATH: &str = "./data/arxiv_lance";
const DEFAULT_METADATA_TABLE: &str = "arxiv_metadata";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let database_path =
        env::var("ARXIV_LANCE_DB").unwrap_or_else(|_| DEFAULT_DATABASE_PATH.to_owned());
    let table_name =
        env::var("ARXIV_METADATA_TABLE").unwrap_or_else(|_| DEFAULT_METADATA_TABLE.to_owned());
    let database = connect(&database_path).execute().await?;
    let table = database.open_table(&table_name).execute().await?;
    reset_embedding_columns(&table).await
}

async fn reset_embedding_columns(table: &Table) -> Result<(), Box<dyn std::error::Error>> {
    let columns: Vec<String> = table
        .schema()
        .await?
        .fields()
        .iter()
        .map(|field| field.name())
        .filter(|name| name.starts_with("embedding_"))
        .cloned()
        .collect();
    if columns.is_empty() {
        println!("No model-specific embedding columns found; metadata table was unchanged.");
        return Ok(());
    }

    let rows_before = table.count_rows(None).await?;
    let indexes = table.list_indices().await?;
    for index in indexes
        .iter()
        .filter(|index| index.columns.iter().any(|column| columns.contains(column)))
    {
        println!("Dropping index {} on {:?}", index.name, index.columns);
        table.drop_index(&index.name).await?;
    }
    let column_refs = columns.iter().map(String::as_str).collect::<Vec<_>>();
    println!("Dropping embedding columns only: {columns:?}");
    table.drop_columns(&column_refs).await?;

    let rows_after = table.count_rows(None).await?;
    if rows_before != rows_after {
        return Err(format!(
            "metadata row count changed while resetting embeddings: {rows_before} -> {rows_after}"
        )
        .into());
    }
    println!(
        "Reset complete. Metadata table rows preserved: {rows_after}; start abstract indexing to rebuild vectors."
    );
    Ok(())
}
