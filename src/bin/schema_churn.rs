use lancedb::{
    arrow::{
        arrow_array::{
            Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
        },
        arrow_schema::{DataType, Field, Schema},
    },
    connect,
    index::{Index, vector::IvfFlatIndexBuilder},
    query::{ExecutableQuery, QueryBase, Select},
    table::NewColumnTransform,
    DistanceType,
};
use futures::StreamExt;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = connect("memory://").execute().await?;
    let schema = Arc::new(Schema::new(vec![
        Field::new("paper_id", DataType::Utf8, false),
    ]));
    let ids = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(vec!["paper-1", "paper-2", "paper-3"]))],
    )?;
    let table = db
        .create_table("vector_schema_churn", vec![ids])
        .execute()
        .await?;

    for generation in 0..3 {
        let column = format!("embedding_model_{generation}");
        let vector_type = DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            2,
        );
        table
            .add_columns()
            .transform(NewColumnTransform::AllNulls(Arc::new(Schema::new(vec![
                Field::new(&column, vector_type, true),
            ]))))
            .execute()
            .await?;

        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vectors = Arc::new(FixedSizeListArray::try_new(
            item_field,
            2,
            Arc::new(Float32Array::from(vec![
                1.0, 0.0, 0.0, 1.0, 0.7, 0.7,
            ])),
            None,
        )?);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("paper_id", DataType::Utf8, false),
                Field::new(&column, vectors.data_type().clone(), true),
            ])),
            vec![
                Arc::new(StringArray::from(vec!["paper-1", "paper-2", "paper-3"])),
                vectors,
            ],
        )?;
        let mut merge = table.merge_insert(&["paper_id"]);
        merge.when_matched_update_all(None);
        merge.execute(Box::new(RecordBatchIterator::new(
            vec![Ok(batch)].into_iter(),
            Arc::new(Schema::new(vec![
                Field::new("paper_id", DataType::Utf8, false),
                Field::new(
                    &column,
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        2,
                    ),
                    true,
                ),
            ])),
        ))).await?;

        table
            .create_index(
                &[column.as_str()],
                Index::IvfFlat(
                    IvfFlatIndexBuilder::default().distance_type(DistanceType::Cosine),
                ),
            )
            .execute()
            .await?;
        let indexes = table.list_indices().await?;
        let index = indexes
            .iter()
            .find(|index| index.columns.iter().any(|name| name == &column))
            .ok_or_else(|| format!("index on {column} was not created"))?;
        table.drop_index(&index.name).await?;
        table.drop_columns(&[column.as_str()]).await?;
        verify_rows(&table).await?;
        println!("generation {generation}: added vector column, indexed, dropped index, dropped column");
    }
    println!(
        "final columns: {:?}",
        table.schema().await?.fields().iter().map(|field| field.name()).collect::<Vec<_>>()
    );
    verify_rows(&table).await?;
    println!("final data read verified: all 3 paper IDs remain intact");
    Ok(())
}

async fn verify_rows(table: &lancedb::Table) -> Result<(), Box<dyn std::error::Error>> {
    let mut rows = table
        .query()
        .select(Select::columns(&["paper_id"]))
        .execute()
        .await?;
    let mut paper_ids = Vec::new();
    while let Some(batch) = rows.next().await {
        let batch = batch?;
        let ids = batch
            .column_by_name("paper_id")
            .ok_or("paper_id column missing after schema change")?
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("paper_id column is not UTF-8 after schema change")?;
        paper_ids.extend((0..ids.len()).map(|row| ids.value(row).to_owned()));
    }
    paper_ids.sort();
    if paper_ids != ["paper-1", "paper-2", "paper-3"] {
        return Err(format!("paper IDs changed after schema change: {paper_ids:?}").into());
    }
    Ok(())
}
