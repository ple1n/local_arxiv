use axum::{Router, http::HeaderName};
use futures::StreamExt;
use lancedb::{
    Table,
    arrow::arrow_array::{Array, LargeStringArray, StringArray},
    connect,
    index::scalar::FullTextSearchQuery,
    query::{ExecutableQuery, QueryBase, Select},
};
use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerConfig},
    schemars::{self, JsonSchema},
    tool, tool_handler, tool_router,
    transport::{
        StreamableHttpService, streamable_http_server::session::local::LocalSessionManager,
    },
};
use serde::{Deserialize, Serialize};
use std::{env, error::Error, net::SocketAddr, sync::Arc, time::Instant};
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
};

const DEFAULT_DATABASE_PATH: &str = "./data/arxiv_lance";
const DEFAULT_TABLE_NAME: &str = "arxiv_full_text";
const DEFAULT_BIND_ADDRESS: &str = "127.0.0.1:8000";
const MAX_SEARCH_RESULTS: usize = 10;
const SEARCH_SNIPPET_CHARS: usize = 600;
const MAX_PAPER_CHARS: usize = 1_000_000;

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchArgs {
    #[schemars(description = "Words or phrase to find in indexed paper text")]
    query: String,
    #[schemars(description = "Maximum results to return, from 1 to 10; defaults to 5")]
    limit: Option<usize>,
    #[schemars(description = "Number of matching results to skip; defaults to 0")]
    offset: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct PaperArgs {
    #[schemars(description = "arXiv paper identifier, for example 1706.03762")]
    paper_id: String,
    #[schemars(description = "Maximum text characters to return; defaults to 12000")]
    max_chars: Option<usize>,
}

#[derive(Debug, Serialize)]
struct SearchHit {
    paper_id: String,
    metadata: Option<PaperMetadata>,
    snippet: String,
    snippet_truncated: bool,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<SearchHit>,
    metadata: SearchMetadata,
}

#[derive(Debug, Serialize)]
struct SearchMetadata {
    lookup_time_ms: u64,
    result_count: usize,
    offset: usize,
    next_offset: usize,
    has_more: bool,
}

#[derive(Debug, Serialize)]
struct PaperMetadata {
    title: Option<String>,
    authors: Option<String>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
    categories: Option<String>,
    primary_category: Option<String>,
    doi: Option<String>,
    journal_ref: Option<String>,
    license: Option<String>,
}

#[derive(Clone)]
struct ArxivMcpServer {
    table: Table,
    metadata_table: Table,
    table_name: String,
    row_count: usize,
    metadata_row_count: usize,
    database_path: String,
    #[expect(dead_code, reason = "tool_handler macro reads this router")]
    tool_router: ToolRouter<Self>,
}

impl ArxivMcpServer {
    fn new(
        table: Table,
        metadata_table: Table,
        table_name: String,
        row_count: usize,
        metadata_row_count: usize,
        database_path: String,
    ) -> Self {
        Self {
            table,
            metadata_table,
            table_name,
            row_count,
            metadata_row_count,
            database_path,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl ArxivMcpServer {
    #[tool(
        description = "Search the local arXiv full-text index. Returns matching paper IDs and short text snippets."
    )]
    async fn search_full_text(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<String, McpError> {
        let started_at = Instant::now();
        let query = args.query.trim();
        if query.is_empty() {
            return Err(McpError::invalid_params("query must not be empty", None));
        }
        if query.len() > 512 {
            return Err(McpError::invalid_params(
                "query must be at most 512 bytes",
                None,
            ));
        }
        if args.limit == Some(0) {
            return Err(McpError::invalid_params("limit must be at least 1", None));
        }
        let limit = args.limit.unwrap_or(5).min(MAX_SEARCH_RESULTS);
        let offset = args.offset.unwrap_or(0);
        let query_limit = limit
            .checked_add(1)
            .ok_or_else(|| McpError::invalid_params("limit is too large", None))?;
        let mut stream = self
            .table
            .query()
            .full_text_search(FullTextSearchQuery::new(query.to_owned()))
            .select(Select::columns(&["paper_id", "text"]))
            .limit(query_limit)
            .offset(offset)
            .execute()
            .await
            .map_err(internal_error)?;

        let mut hits = Vec::new();
        let mut has_more = false;
        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(internal_error)?;
            for row in 0..batch.num_rows() {
                if hits.len() == limit {
                    has_more = true;
                    break;
                }
                let paper_id = string_value(batch.column(0).as_ref(), row)
                    .map_err(internal_error)?
                    .unwrap_or_default();
                let text = string_value(batch.column(1).as_ref(), row)
                    .map_err(internal_error)?
                    .unwrap_or_default();
                let (snippet, snippet_truncated) =
                    search_excerpt(text, query, SEARCH_SNIPPET_CHARS);
                let paper_id = paper_id.to_owned();
                let metadata = self.metadata_for_id(&paper_id).await?;
                hits.push(SearchHit {
                    paper_id,
                    metadata,
                    snippet,
                    snippet_truncated,
                });
            }
            if has_more {
                break;
            }
        }
        let next_offset = offset
            .checked_add(hits.len())
            .ok_or_else(|| McpError::invalid_params("offset is too large", None))?;
        let response = SearchResponse {
            metadata: SearchMetadata {
                lookup_time_ms: started_at.elapsed().as_millis() as u64,
                result_count: hits.len(),
                offset,
                next_offset,
                has_more,
            },
            results: hits,
        };
        serde_json::to_string_pretty(&response).map_err(internal_error)
    }

    #[tool(description = "Retrieve the stored text of a paper by its arXiv paper_id.")]
    async fn get_paper_text(
        &self,
        Parameters(args): Parameters<PaperArgs>,
    ) -> Result<String, McpError> {
        let paper_id = args.paper_id.trim();
        if paper_id.is_empty() || paper_id.len() > 128 {
            return Err(McpError::invalid_params(
                "paper_id must contain between 1 and 128 bytes",
                None,
            ));
        }
        let max_chars = args.max_chars.unwrap_or(MAX_PAPER_CHARS);
        if max_chars == 0 {
            return Err(McpError::invalid_params(
                "max_chars must be at least 1",
                None,
            ));
        }
        let max_chars = max_chars.min(MAX_PAPER_CHARS);
        let escaped_id = paper_id.replace('\'', "''");
        let mut stream = self
            .table
            .query()
            .only_if(format!("paper_id = '{escaped_id}'"))
            .select(Select::columns(&["paper_id", "text"]))
            .limit(1)
            .execute()
            .await
            .map_err(internal_error)?;

        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(internal_error)?;
            if batch.num_rows() == 0 {
                continue;
            }
            let id = string_value(batch.column(0).as_ref(), 0)
                .map_err(internal_error)?
                .unwrap_or_default();
            let text = string_value(batch.column(1).as_ref(), 0)
                .map_err(internal_error)?
                .unwrap_or_default();
            let (text, truncated) = truncate_chars(text, max_chars);
            let metadata = self.metadata_for_id(id).await?;
            return serde_json::to_string_pretty(&serde_json::json!({
                "paper_id": id,
                "metadata": metadata,
                "text": text,
                "truncated": truncated,
                "returned_chars": text.chars().count(),
                "max_chars": max_chars,
            }))
            .map_err(internal_error);
        }

        Ok(format!(
            "No paper found with paper_id {paper_id:?} in this table."
        ))
    }

    async fn metadata_for_id(&self, paper_id: &str) -> Result<Option<PaperMetadata>, McpError> {
        let escaped_id = paper_id.replace('\'', "''");
        let mut stream = self
            .metadata_table
            .query()
            .only_if(format!("paper_id = '{escaped_id}'"))
            .select(Select::columns(&[
                "title",
                "authors",
                "abstract",
                "categories",
                "primary_category",
                "doi",
                "journal_ref",
                "license",
            ]))
            .limit(1)
            .execute()
            .await
            .map_err(internal_error)?;
        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(internal_error)?;
            if batch.num_rows() == 0 {
                continue;
            }
            let column = |index| -> Result<Option<String>, McpError> {
                string_value(batch.column(index).as_ref(), 0)
                    .map(|value| value.map(str::to_owned))
                    .map_err(internal_error)
            };
            return Ok(Some(PaperMetadata {
                title: column(0)?,
                authors: column(1)?,
                abstract_text: column(2)?,
                categories: column(3)?,
                primary_category: column(4)?,
                doi: column(5)?,
                journal_ref: column(6)?,
                license: column(7)?,
            }));
        }
        Ok(None)
    }

    #[tool(description = "Report the local arXiv table location and number of indexed papers.")]
    fn dataset_status(&self) -> String {
        serde_json::to_string_pretty(&serde_json::json!({
            "database_path": self.database_path,
            "table": self.table_name,
            "text_rows": self.row_count,
            "metadata_rows": self.metadata_row_count,
            "search_column": "text",
            "metadata_columns": [
                "title", "authors", "abstract", "categories",
                "primary_category", "doi", "journal_ref", "license"
            ],
            "scope": "This reports only the data currently loaded in this LanceDB table."
        }))
        .expect("serializing a JSON value cannot fail")
    }
}

#[tool_handler]
impl ServerHandler for ArxivMcpServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Search the locally ingested arXiv paper text with full-text search. \
                 The current table may be a sample rather than the complete arXiv dataset.",
        )
    }
}

pub async fn run() -> Result<(), Box<dyn Error>> {
    let database_path =
        env::var("ARXIV_LANCE_DB").unwrap_or_else(|_| DEFAULT_DATABASE_PATH.to_owned());
    let table_name =
        env::var("ARXIV_LANCE_TABLE").unwrap_or_else(|_| DEFAULT_TABLE_NAME.to_owned());
    let bind_address: SocketAddr = env::var("MCP_ADDR")
        .unwrap_or_else(|_| DEFAULT_BIND_ADDRESS.to_owned())
        .parse()?;

    let database = connect(&database_path).execute().await?;
    let table = database.open_table(&table_name).execute().await?;
    let indices = table.list_indices().await?;
    if !indices.iter().any(|index| {
        index.index_type.to_string() == "FTS" && index.columns.iter().any(|column| column == "text")
    }) {
        return Err(
            format!("Table {table_name:?} has no full-text index on the text column").into(),
        );
    }
    let row_count = table.count_rows(None).await?;
    let metadata_table_name =
        env::var("ARXIV_METADATA_TABLE").unwrap_or_else(|_| "arxiv_metadata".to_owned());
    let metadata_table = database.open_table(&metadata_table_name).execute().await?;
    let metadata_indices = metadata_table.list_indices().await?;
    if !metadata_indices.iter().any(|index| {
        index.index_type.to_string() == "BTREE"
            && index.columns.iter().any(|column| column == "paper_id")
    }) {
        return Err(format!(
            "Metadata table {metadata_table_name:?} has no B-tree index on paper_id"
        )
        .into());
    }
    let metadata_row_count = metadata_table.count_rows(None).await?;
    let server = ArxivMcpServer::new(
        table,
        metadata_table,
        table_name,
        row_count,
        metadata_row_count,
        database_path.clone(),
    );

    let service: StreamableHttpService<ArxivMcpServer, LocalSessionManager> =
        StreamableHttpService::new(
            move || Ok(server.clone()),
            Arc::new(LocalSessionManager::default()),
            Default::default(),
        );
    let router = Router::new()
        .route("/health", axum::routing::get(|| async { "ok" }))
        .nest_service("/mcp", service)
        .fallback_service(ServeDir::new("web/dist").append_index_html_on_directories(true))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
                .expose_headers([HeaderName::from_static("mcp-session-id")]),
        );
    let listener = tokio::net::TcpListener::bind(bind_address).await?;
    println!(
        "ArXiv MCP server listening at http://{}/mcp (database: {database_path})",
        listener.local_addr()?
    );
    axum::serve(listener, router).await?;
    Ok(())
}

fn string_value(array: &dyn Array, row: usize) -> Result<Option<&str>, String> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(Some(values.value(row)));
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(Some(values.value(row)));
    }
    Err(format!(
        "Expected a UTF-8 string column, got {:?}",
        array.data_type()
    ))
}

fn truncate_chars(value: &str, max_chars: usize) -> (String, bool) {
    match value.char_indices().nth(max_chars) {
        Some((byte_index, _)) => (value[..byte_index].to_owned(), true),
        None => (value.to_owned(), false),
    }
}

fn search_excerpt(text: &str, query: &str, max_chars: usize) -> (String, bool) {
    let normalized_text = text.to_ascii_lowercase();
    let phrase = query.to_ascii_lowercase();
    let match_byte = normalized_text.find(&phrase).or_else(|| {
        query
            .split(|character: char| !character.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .max_by_key(|word| word.chars().count())
            .and_then(|word| normalized_text.find(&word.to_ascii_lowercase()))
    });
    let Some(match_byte) = match_byte else {
        return truncate_chars(text, max_chars);
    };

    let match_char = text[..match_byte].chars().count();
    let start_char = match_char.saturating_sub(max_chars / 3);
    let start_byte = text
        .char_indices()
        .nth(start_char)
        .map(|(byte, _)| byte)
        .unwrap_or(0);
    let (excerpt, truncated) = truncate_chars(&text[start_byte..], max_chars);
    let excerpt = if start_char > 0 {
        format!("…{excerpt}")
    } else {
        excerpt
    };
    let excerpt = if truncated {
        format!("{excerpt}…")
    } else {
        excerpt
    };
    (excerpt, start_char > 0 || truncated)
}

fn internal_error(error: impl std::fmt::Display) -> McpError {
    McpError::internal_error(error.to_string(), None)
}

#[cfg(test)]
mod tests {
    use super::{SEARCH_SNIPPET_CHARS, search_excerpt};

    #[test]
    fn excerpt_centers_on_exact_query_phrase() {
        let text = format!(
            "{}The Authors reply to the Comment of Golub and Lamoreaux.{}",
            "x".repeat(1000),
            "y".repeat(1000)
        );
        let (excerpt, truncated) = search_excerpt(
            &text,
            "The Authors reply to the Comment of Golub and Lamoreaux",
            SEARCH_SNIPPET_CHARS,
        );

        assert!(truncated);
        assert!(excerpt.contains("The Authors reply to the Comment of Golub and Lamoreaux"));
        assert!(!excerpt.starts_with(&"x".repeat(30)));
    }
}
