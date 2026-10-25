use crate::test_pipeline::Embedder;
use axum::{
    Json, Router,
    extract::Path,
    http::HeaderName,
    response::sse::{Event, KeepAlive, Sse},
    routing::get,
};
use futures::{StreamExt, stream};
use lancedb::{
    Table,
    arrow::arrow_array::{Array, BooleanArray, LargeStringArray, StringArray},
    connect,
    index::{
        Index,
        scalar::{BTreeIndexBuilder, BitmapIndexBuilder, FullTextSearchQuery},
    },
    query::{ExecutableQuery, QueryBase, Select},
    table::NewColumnTransform,
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
use std::{
    env,
    error::Error,
    net::{IpAddr, SocketAddr},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;
use tokio::sync::{Notify, RwLock, Semaphore, broadcast};
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
};

const DEFAULT_DATABASE_PATH: &str = "./data/arxiv_lance";
const DEFAULT_TABLE_NAME: &str = "arxiv_full_text";
const MAX_SEARCH_RESULTS: usize = 10;
const SEARCH_SNIPPET_CHARS: usize = 600;
const MAX_PAPER_CHARS: usize = 1_000_000;
const DEFAULT_EMBEDDING_CTX: usize = 32_768;
const MAX_LFM_EMBEDDING_TOKENS: usize = 512;
const DEFAULT_CHUNK_TOKEN_BUDGET: usize =
    MAX_LFM_EMBEDDING_TOKENS - crate::test_pipeline::EMBEDDING_TOKEN_HEADROOM;

static NEXT_INDEX_JOB_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Serialize)]
struct IndexJobProgress {
    job_id: String,
    state: String,
    total_papers: usize,
    scanned_papers: usize,
    elapsed_seconds: f64,
    papers_per_second: f64,
    eta_seconds: Option<f64>,
    chunked_papers: usize,
    embedded_papers: usize,
    failed_papers: usize,
    current_paper: Option<String>,
    phase: Option<String>,
    error: Option<String>,
    #[serde(skip_serializing)]
    started_at: Instant,
}

#[derive(Clone)]
struct IndexJobManager {
    progress: Arc<RwLock<Option<IndexJobProgress>>>,
    updates: broadcast::Sender<IndexJobProgress>,
    active: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    resume: Arc<Notify>,
}

impl IndexJobManager {
    fn new() -> Self {
        let (updates, _) = broadcast::channel(32);
        Self {
            progress: Arc::new(RwLock::new(None)),
            updates,
            active: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
            resume: Arc::new(Notify::new()),
        }
    }

    async fn start(&self, paper_id: Option<String>, total_papers: usize) -> Result<String, String> {
        if self
            .active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err("An indexing job is already running".to_owned());
        }
        let job_id = format!(
            "{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_millis(),
            NEXT_INDEX_JOB_ID.fetch_add(1, Ordering::Relaxed)
        );
        let initial = IndexJobProgress {
            job_id: job_id.clone(),
            state: "queued".to_owned(),
            total_papers,
            scanned_papers: 0,
            elapsed_seconds: 0.0,
            papers_per_second: 0.0,
            eta_seconds: None,
            chunked_papers: 0,
            embedded_papers: 0,
            failed_papers: 0,
            current_paper: paper_id,
            phase: Some("queued".to_owned()),
            error: None,
            started_at: Instant::now(),
        };
        self.paused.store(false, Ordering::Release);
        *self.progress.write().await = Some(initial.clone());
        let _ = self.updates.send(initial);
        Ok(job_id)
    }

    async fn update(&self, update: impl FnOnce(&mut IndexJobProgress)) {
        if let Some(progress) = self.progress.write().await.as_mut() {
            update(progress);
            progress.elapsed_seconds = progress.started_at.elapsed().as_secs_f64();
            progress.papers_per_second = if progress.elapsed_seconds > 0.0 {
                progress.scanned_papers as f64 / progress.elapsed_seconds
            } else {
                0.0
            };
            progress.eta_seconds = if progress.papers_per_second > 0.0 {
                Some(
                    progress
                        .total_papers
                        .saturating_sub(progress.scanned_papers) as f64
                        / progress.papers_per_second,
                )
            } else {
                None
            };
            let _ = self.updates.send(progress.clone());
        }
    }

    async fn finish(&self, state: &str, error: Option<String>) {
        self.update(|progress| {
            progress.state = state.to_owned();
            progress.current_paper = None;
            progress.phase = None;
            progress.error = error;
        })
        .await;
        self.paused.store(false, Ordering::Release);
        self.resume.notify_waiters();
        self.resume.notify_one();
        self.active.store(false, Ordering::Release);
    }

    async fn set_paused(&self, job_id: &str, paused: bool) -> Result<(), String> {
        let mut current = self.progress.write().await;
        let progress = current
            .as_mut()
            .filter(|progress| progress.job_id == job_id)
            .ok_or_else(|| format!("No indexing job found with ID {job_id:?}"))?;
        if !matches!(progress.state.as_str(), "running" | "paused") {
            return Err(format!(
                "Cannot change pause state while indexing job is {}",
                progress.state
            ));
        }
        self.paused.store(paused, Ordering::Release);
        progress.state = if paused { "paused" } else { "running" }.to_owned();
        let snapshot = progress.clone();
        drop(current);
        let _ = self.updates.send(snapshot);
        if !paused {
            self.resume.notify_waiters();
            self.resume.notify_one();
        }
        Ok(())
    }

    async fn wait_until_resumed(&self) {
        loop {
            let resumed = self.resume.notified();
            if !self.paused.load(Ordering::Acquire) {
                return;
            }
            resumed.await;
        }
    }

    async fn subscribe(
        &self,
        job_id: &str,
    ) -> Result<(IndexJobProgress, broadcast::Receiver<IndexJobProgress>), String> {
        let receiver = self.updates.subscribe();
        let current = self
            .progress
            .read()
            .await
            .clone()
            .filter(|progress| progress.job_id == job_id)
            .ok_or_else(|| format!("No indexing job found with ID {job_id:?}"))?;
        Ok((current, receiver))
    }
}

struct JobPausableEmbedder<'a> {
    inner: &'a dyn Embedder,
    jobs: Option<&'a IndexJobManager>,
}

impl Embedder for JobPausableEmbedder<'_> {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        task: crate::test_pipeline::EmbeddingTask,
        ctx: usize,
    ) -> crate::test_pipeline::ModelFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            if let Some(jobs) = self.jobs {
                jobs.wait_until_resumed().await;
            }
            self.inner.embed(texts, task, ctx).await
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AppConfig {
    backend: ListenerConfig,
    frontend: ListenerConfig,
    embedding: ListenerConfig,
    llama: ListenerConfig,
    #[serde(skip_serializing)]
    models: ModelsConfig,
    #[serde(rename = "apiBaseUrl")]
    api_base_url: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelsConfig {
    embedding: EmbeddingModelConfig,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EmbeddingModelConfig {
    id: String,
    repo: String,
    path: String,
    tokenizer_path: String,
    max_tokens: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ListenerConfig {
    host: IpAddr,
    port: u16,
}

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

#[derive(Debug, Deserialize, JsonSchema)]
struct ChunkArgs {
    #[schemars(description = "arXiv paper identifier, for example 1706.03762")]
    paper_id: String,
    #[schemars(
        description = "Tokenizer-measured budget per chunk; defaults to 480 and is capped by the safe model limit"
    )]
    token_budget: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EmbedChunkBatchArgs {
    #[schemars(description = "LanceDB table containing the chunks to embed")]
    chunk_table: String,
    #[schemars(description = "Chunk IDs from chunk_paper, 1 to 8 per embedding request")]
    chunk_ids: Vec<String>,
    #[schemars(description = "Embedding model ID from list_models")]
    model: String,
    #[schemars(
        description = "Embedding context; defaults to 32768 and applies a 32-token safety margin below the configured model limit"
    )]
    ctx: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct FinalizeChunkEmbeddingsArgs {
    #[schemars(description = "LanceDB table containing the chunk vectors")]
    chunk_table: String,
    #[schemars(description = "Embedding model ID whose vector column should be indexed")]
    model: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct TestPaperArgs {
    #[schemars(
        description = "arXiv paper identifier whose retained test chunk table should be searched"
    )]
    paper_id: String,
    #[schemars(
        description = "Optional embedding model ID from list_models; defaults to configured model"
    )]
    embedding_model: Option<String>,
    #[schemars(description = "Optional reranking model ID from list_models")]
    reranker_model: Option<String>,
    #[schemars(description = "Query for shared lexical, dense, fusion, and reranking search")]
    query: String,
    #[schemars(
        description = "Maximum hybrid candidates returned and optionally reranked; defaults to 30, maximum 100"
    )]
    candidate_limit: Option<usize>,
    #[schemars(
        description = "Embedding context; defaults to 32768 and applies a 32-token safety margin below the configured model limit"
    )]
    ctx: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EmbedTextsArgs {
    #[schemars(description = "Non-empty text passages to embed; provide 1 to 8 passages")]
    texts: Vec<String>,
    #[schemars(description = "Optional embedding model ID from list_models")]
    model: Option<String>,
    #[schemars(
        description = "Embedding context; defaults to 32768 and applies a 32-token safety margin below the configured model limit"
    )]
    ctx: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct HybridChunkSearchArgs {
    #[schemars(
        description = "LanceDB chunk table containing indexed text and model-specific vector columns"
    )]
    chunk_table: String,
    #[schemars(description = "Text query to retrieve and fuse lexical and dense chunk candidates")]
    query: String,
    #[schemars(description = "Embedding model ID whose model-specific vector column is searched")]
    embedding_model: String,
    #[schemars(description = "Optional reranking model ID")]
    reranker_model: Option<String>,
    #[schemars(description = "Maximum fused candidates to return; defaults to 30, maximum 100")]
    candidate_limit: Option<usize>,
    #[schemars(
        description = "Embedding context; defaults to 32768 and applies a 32-token safety margin below the configured model limit"
    )]
    ctx: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct StartIndexJobArgs {
    #[schemars(
        description = "Optional paper ID for a one-paper run; omit to process all pending papers"
    )]
    paper_id: Option<String>,
    #[schemars(
        description = "Tokenizer-measured chunk budget; defaults to 480 and is capped by the safe model limit"
    )]
    token_budget: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SetIndexJobPausedArgs {
    #[schemars(description = "Index job ID returned by start_indexing_job")]
    job_id: String,
    #[schemars(description = "Pause after the in-flight embedding request finishes, or resume")]
    paused: bool,
}

#[derive(Debug, Deserialize)]
struct ModelListResponse {
    data: Vec<ModelListItem>,
}

#[derive(Debug, Deserialize)]
struct ModelListItem {
    id: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    status: Option<ModelListStatus>,
}

#[derive(Debug, Deserialize)]
struct ModelListStatus {
    #[serde(default)]
    args: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AvailableModel {
    id: String,
    tags: Vec<String>,
}

#[derive(Debug, Serialize)]
struct AvailableModelsResponse {
    embedding: Vec<AvailableModel>,
    reranking: Vec<AvailableModel>,
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
    search_url: String,
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
    embedding_model: String,
    embedding_endpoint: String,
    model_list_endpoint: String,
    reranker_endpoint: String,
    embedding_tokenizer: Arc<Tokenizer>,
    embedding_scheduler: Arc<Semaphore>,
    embedding_ctx: usize,
    embedding_max_tokens: usize,
    index_jobs: IndexJobManager,
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
        embedding: EmbeddingModelConfig,
        embedding_endpoint: String,
        model_list_endpoint: String,
        reranker_endpoint: String,
        embedding_tokenizer: Arc<Tokenizer>,
        embedding_scheduler: Arc<Semaphore>,
        index_jobs: IndexJobManager,
    ) -> Self {
        Self {
            table,
            metadata_table,
            table_name,
            row_count,
            metadata_row_count,
            database_path,
            embedding_model: embedding.id,
            embedding_endpoint,
            model_list_endpoint,
            reranker_endpoint,
            embedding_tokenizer,
            embedding_scheduler,
            embedding_ctx: DEFAULT_EMBEDDING_CTX,
            embedding_max_tokens: embedding.max_tokens,
            index_jobs,
            tool_router: Self::tool_router(),
        }
    }

    fn embedder(&self, model: &str) -> Result<crate::test_pipeline::HttpEmbedder, McpError> {
        crate::test_pipeline::HttpEmbedder::new(
            model,
            &self.embedding_endpoint,
            self.embedding_tokenizer.clone(),
            self.embedding_max_tokens,
            self.embedding_scheduler.clone(),
        )
        .map_err(internal_error)
    }

    fn chunk_embedding_text(
        &self,
        text: &str,
        requested_budget: usize,
        ctx: usize,
    ) -> Result<Vec<crate::chunking::Chunk>, Box<dyn Error + Send + Sync>> {
        let safe_limit =
            crate::test_pipeline::effective_embedding_token_limit(ctx, self.embedding_max_tokens)?;
        let budget = requested_budget.min(safe_limit);
        let tokenization_error = std::cell::RefCell::new(None);
        let cost = |chunk: &str| {
            let input = format!("document: {chunk}");
            match self.embedding_tokenizer.encode(input, true) {
                Ok(encoding) => encoding.get_ids().len(),
                Err(error) => {
                    *tokenization_error.borrow_mut() = Some(error.to_string());
                    usize::MAX
                }
            }
        };
        let chunks = crate::chunking::chunk_text_with_cost(text, budget, &cost);
        if let Some(error) = tokenization_error.into_inner() {
            return Err(error.into());
        }
        if chunks.iter().any(|chunk| chunk.cost > safe_limit) {
            return Err("tokenizer could not split text within the safe embedding limit".into());
        }
        Ok(chunks)
    }

    fn reranker(&self, model: &str) -> Result<crate::test_pipeline::HttpReranker, McpError> {
        crate::test_pipeline::HttpReranker::from_endpoint_with_model(
            Some(model),
            &self.reranker_endpoint,
        )
        .map_err(internal_error)
    }

    async fn run_index_job(
        &self,
        paper_id: Option<String>,
        token_budget: usize,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let database = connect(&self.database_path).execute().await?;
        let filter = if let Some(paper_id) = &paper_id {
            format!("paper_id = {}", sql_string(paper_id))
        } else {
            pending_index_filter(&self.embedding_model)
        };
        let mut rows = self
            .table
            .query()
            .only_if(filter)
            .select(Select::columns(&[
                "paper_id",
                "text",
                "chunked",
                "embedded",
                "embedding_model",
            ]))
            .execute()
            .await?;
        let mut scanned = 0;
        let mut chunked_count = 0;
        let mut embedded_count = 0;
        let mut failed_count = 0;
        let mut last_progress_update = Instant::now();
        while let Some(batch) = rows.next().await {
            let batch = batch?;
            for row in 0..batch.num_rows() {
                self.index_jobs.wait_until_resumed().await;
                let current_id = string_at(&batch, "paper_id", row)?
                    .ok_or("indexing source row has a null paper_id")?;
                let text = string_at(&batch, "text", row)?
                    .ok_or_else(|| format!("paper {current_id:?} has null text"))?;
                let chunked = optional_bool_value(
                    batch
                        .column_by_name("chunked")
                        .ok_or("missing chunked state")?
                        .as_ref(),
                    row,
                )?;
                let embedded = optional_bool_value(
                    batch
                        .column_by_name("embedded")
                        .ok_or("missing embedded state")?
                        .as_ref(),
                    row,
                )?;
                let indexed_model = string_at(&batch, "embedding_model", row)?;
                self.index_jobs
                    .update(|progress| {
                        progress.current_paper = Some(current_id.clone());
                        progress.phase = Some("processing".to_owned());
                    })
                    .await;
                match self
                    .process_paper(
                        &database,
                        &current_id,
                        &text,
                        chunked,
                        embedded,
                        indexed_model.as_deref(),
                        self.embedding_ctx,
                        token_budget,
                        paper_id.is_some(),
                        true,
                    )
                    .await
                {
                    Ok((_, true)) => {
                        if chunked != Some(true) {
                            chunked_count += 1;
                        }
                        embedded_count += 1;
                    }
                    Ok((_, false)) => {}
                    Err(error) => {
                        failed_count += 1;
                        update_paper_index_state(
                            &self.table,
                            &current_id,
                            &[("indexing_error", sql_string(&error.to_string()))],
                        )
                        .await?;
                    }
                }
                scanned += 1;
                if scanned % 100 == 0
                    || paper_id.is_some()
                    || last_progress_update.elapsed().as_secs_f64() >= 1.0
                {
                    self.index_jobs
                        .update(|progress| {
                            progress.scanned_papers = scanned;
                            progress.chunked_papers = chunked_count;
                            progress.embedded_papers = embedded_count;
                            progress.failed_papers = failed_count;
                            progress.current_paper = Some(current_id);
                        })
                        .await;
                    last_progress_update = Instant::now();
                }
            }
        }
        if paper_id.is_some() && scanned == 0 {
            return Err(format!(
                "No paper found with paper_id {:?}",
                paper_id.as_deref().unwrap_or_default()
            )
            .into());
        }
        self.index_jobs
            .update(|progress| {
                progress.scanned_papers = scanned;
                progress.chunked_papers = chunked_count;
                progress.embedded_papers = embedded_count;
                progress.failed_papers = failed_count;
            })
            .await;
        if embedded_count > 0 {
            let pipeline = crate::hybrid_pipeline::HybridPipeline::new(
                &database,
                crate::hybrid_pipeline::TableNames {
                    chunks: crate::hybrid_pipeline::GLOBAL_CHUNK_TABLE.to_owned(),
                },
            );
            pipeline.ensure_vector_index(&self.embedding_model).await?;
        }
        Ok(())
    }

    async fn index_one_paper(
        &self,
        paper_id: &str,
        ctx: usize,
        token_budget: usize,
    ) -> Result<(usize, bool), Box<dyn Error + Send + Sync>> {
        let escaped_id = paper_id.replace('\'', "''");
        let mut rows = self
            .table
            .query()
            .only_if(format!("paper_id = '{escaped_id}'"))
            .select(Select::columns(&[
                "paper_id",
                "text",
                "chunked",
                "embedded",
                "embedding_model",
            ]))
            .limit(1)
            .execute()
            .await?;
        while let Some(batch) = rows.next().await {
            let batch = batch?;
            if batch.num_rows() == 0 {
                continue;
            }
            let id = string_at(&batch, "paper_id", 0)?
                .ok_or("indexing source row has a null paper_id")?;
            let text = string_at(&batch, "text", 0)?
                .ok_or_else(|| format!("paper {id:?} has null text"))?;
            let chunked = optional_bool_value(
                batch
                    .column_by_name("chunked")
                    .ok_or("missing chunked state")?
                    .as_ref(),
                0,
            )?;
            let embedded = optional_bool_value(
                batch
                    .column_by_name("embedded")
                    .ok_or("missing embedded state")?
                    .as_ref(),
                0,
            )?;
            let model = string_at(&batch, "embedding_model", 0)?;
            let database = connect(&self.database_path).execute().await?;
            return self
                .process_paper(
                    &database,
                    &id,
                    &text,
                    chunked,
                    embedded,
                    model.as_deref(),
                    ctx,
                    token_budget,
                    true,
                    false,
                )
                .await;
        }
        Err(format!("No paper found with paper_id {paper_id:?}").into())
    }

    async fn process_paper(
        &self,
        database: &lancedb::Connection,
        paper_id: &str,
        text: &str,
        chunked: Option<bool>,
        embedded: Option<bool>,
        indexed_model: Option<&str>,
        ctx: usize,
        token_budget: usize,
        finalize_index: bool,
        pauseable: bool,
    ) -> Result<(usize, bool), Box<dyn Error + Send + Sync>> {
        let pipeline = crate::hybrid_pipeline::HybridPipeline::new(
            database,
            crate::hybrid_pipeline::TableNames {
                chunks: crate::hybrid_pipeline::GLOBAL_CHUNK_TABLE.to_owned(),
            },
        );
        self.index_jobs
            .update(|progress| {
                progress.current_paper = Some(paper_id.to_owned());
                progress.phase = Some("checking chunks".to_owned());
            })
            .await;
        let same_model = indexed_model == Some(self.embedding_model.as_str());
        if chunked != Some(true) {
            self.index_jobs
                .update(|progress| progress.phase = Some("tokenizing and chunking".to_owned()))
                .await;
            let chunks = self.chunk_embedding_text(text, token_budget, ctx)?;
            if chunks.is_empty() {
                return Err(format!("paper {paper_id:?} produced no chunks").into());
            }
            pipeline.upsert_chunks(paper_id, &chunks).await?;
            let chunk_count = chunks.len();
            update_paper_index_state(
                &self.table,
                paper_id,
                &[
                    ("chunked", "true".to_owned()),
                    ("embedded", "false".to_owned()),
                    ("embedding_model", sql_string(&self.embedding_model)),
                    ("chunk_count", chunk_count.to_string()),
                    ("embedded_chunk_count", "0".to_owned()),
                    ("indexing_error", "NULL".to_owned()),
                ],
            )
            .await?;
        } else if embedded == Some(true) && same_model {
            let (total, _) = pipeline
                .paper_embedding_progress(paper_id, &self.embedding_model)
                .await?;
            return Ok((total, true));
        } else if !same_model {
            update_paper_index_state(
                &self.table,
                paper_id,
                &[
                    ("embedded", "false".to_owned()),
                    ("embedding_model", sql_string(&self.embedding_model)),
                    ("embedded_chunk_count", "0".to_owned()),
                    ("indexing_error", "NULL".to_owned()),
                ],
            )
            .await?;
        }

        let embedder = self.embedder(&self.embedding_model)?;
        let job_embedder = JobPausableEmbedder {
            inner: &embedder,
            jobs: pauseable.then_some(&self.index_jobs),
        };
        self.index_jobs
            .update(|progress| progress.phase = Some("embedding chunks".to_owned()))
            .await;
        let embedding = if finalize_index {
            pipeline
                .embed_paper_chunks(paper_id, &self.embedding_model, ctx, &job_embedder, 4)
                .await
        } else {
            pipeline
                .embed_paper_chunks_without_index(
                    paper_id,
                    &self.embedding_model,
                    ctx,
                    &job_embedder,
                    4,
                )
                .await
        };
        match embedding {
            Ok(_) => {
                let (chunk_count, embedded_count) = pipeline
                    .paper_embedding_progress(paper_id, &self.embedding_model)
                    .await?;
                update_paper_index_state(
                    &self.table,
                    paper_id,
                    &[
                        ("chunked", "true".to_owned()),
                        ("embedded", "true".to_owned()),
                        ("embedding_model", sql_string(&self.embedding_model)),
                        ("chunk_count", chunk_count.to_string()),
                        ("embedded_chunk_count", embedded_count.to_string()),
                        ("indexing_error", "NULL".to_owned()),
                    ],
                )
                .await?;
                Ok((chunk_count, true))
            }
            Err(error) => {
                let (total, embedded_count) = pipeline
                    .paper_embedding_progress(paper_id, &self.embedding_model)
                    .await?;
                update_paper_index_state(
                    &self.table,
                    paper_id,
                    &[
                        ("chunked", "true".to_owned()),
                        ("embedded", "false".to_owned()),
                        ("embedding_model", sql_string(&self.embedding_model)),
                        ("chunk_count", total.to_string()),
                        ("embedded_chunk_count", embedded_count.to_string()),
                        ("indexing_error", sql_string(&error.to_string())),
                    ],
                )
                .await?;
                Err(error)
            }
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
            search_url: format!(
                "/?tool=search_full_text&query={}&limit={limit}&offset={offset}",
                Self::encode_query_parameter(query)
            ),
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

    fn encode_query_parameter(value: &str) -> String {
        let mut encoded = String::with_capacity(value.len());
        for byte in value.bytes() {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
                encoded.push(char::from(byte));
            } else {
                use std::fmt::Write;
                write!(encoded, "%{byte:02X}").expect("writing to String cannot fail");
            }
        }
        encoded
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

    #[tool(
        description = "Preview how a paper will be chunked. This does not write chunks; start_indexing_job performs durable indexing."
    )]
    async fn chunk_paper(
        &self,
        Parameters(args): Parameters<ChunkArgs>,
    ) -> Result<String, McpError> {
        let paper_id = args.paper_id.trim();
        if paper_id.is_empty() || paper_id.len() > 128 {
            return Err(McpError::invalid_params(
                "paper_id must contain between 1 and 128 bytes",
                None,
            ));
        }
        let token_budget = args.token_budget.unwrap_or(512);
        if token_budget == 0 || token_budget > 2_048 {
            return Err(McpError::invalid_params(
                "token_budget must be between 1 and 2048",
                None,
            ));
        }
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
            let source = string_value(batch.column(1).as_ref(), 0)
                .map_err(internal_error)?
                .unwrap_or_default();
            let (source, truncated) = truncate_chars(source, MAX_PAPER_CHARS);
            let table_name = crate::hybrid_pipeline::GLOBAL_CHUNK_TABLE;
            let chunks = self
                .chunk_embedding_text(&source, token_budget, self.embedding_ctx)
                .map_err(internal_error)?;
            return serde_json::to_string_pretty(&serde_json::json!({
                "paper_id": id,
                "chunk_table": table_name,
                "token_budget": token_budget,
                "source_chars": source.chars().count(),
                "source_truncated": truncated,
                "chunks": chunks.iter().map(|chunk| serde_json::json!({
                    "chunk_id": crate::hybrid_pipeline::HybridPipeline::chunk_id(paper_id, chunk),
                    "text": chunk.text,
                    "byte_range": chunk.byte_range,
                    "word_count": chunk.word_count,
                    "cost": chunk.cost,
                })).collect::<Vec<_>>(),
            }))
            .map_err(internal_error);
        }

        Err(McpError::invalid_params(
            format!("No paper found with paper_id {paper_id:?} in this table."),
            None,
        ))
    }

    #[tool(
        description = "Queue indexing for one paper, or all papers not yet chunked and embedded. Progress is streamed from the backend."
    )]
    async fn start_indexing_job(
        &self,
        Parameters(args): Parameters<StartIndexJobArgs>,
    ) -> Result<String, McpError> {
        let paper_id = args.paper_id.map(|value| value.trim().to_owned());
        if paper_id
            .as_deref()
            .is_some_and(|id| id.is_empty() || id.len() > 128)
        {
            return Err(McpError::invalid_params(
                "paper_id must contain between 1 and 128 bytes",
                None,
            ));
        }
        let token_budget = args.token_budget.unwrap_or(DEFAULT_CHUNK_TOKEN_BUDGET);
        if token_budget == 0 || token_budget > 2_048 {
            return Err(McpError::invalid_params(
                "token_budget must be between 1 and 2048",
                None,
            ));
        }
        let total_papers = if paper_id.is_some() {
            self.table
                .count_rows(Some(format!(
                    "paper_id = {}",
                    sql_string(paper_id.as_deref().unwrap_or_default())
                )))
                .await
                .map_err(internal_error)?
        } else {
            self.table
                .count_rows(Some(pending_index_filter(&self.embedding_model)))
                .await
                .map_err(internal_error)?
        };
        if total_papers == 0 {
            return Err(McpError::invalid_params(
                "There are no papers pending indexing",
                None,
            ));
        }
        let job_id = self
            .index_jobs
            .start(paper_id.clone(), total_papers)
            .await
            .map_err(|error| McpError::invalid_request(error, None))?;
        let server = self.clone();
        let jobs = self.index_jobs.clone();
        let running_id = job_id.clone();
        tokio::spawn(async move {
            jobs.update(|progress| progress.state = "running".to_owned())
                .await;
            let heartbeat_jobs = jobs.clone();
            let heartbeat = tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(1));
                loop {
                    interval.tick().await;
                    heartbeat_jobs.update(|_| {}).await;
                }
            });
            match server.run_index_job(paper_id, token_budget).await {
                Ok(()) => jobs.finish("complete", None).await,
                Err(error) => jobs.finish("failed", Some(error.to_string())).await,
            };
            heartbeat.abort();
        });
        Ok(serde_json::json!({
            "job_id": running_id,
            "events_url": format!("/indexing-jobs/{running_id}/events"),
            "state": "queued",
        })
        .to_string())
    }

    #[tool(
        description = "Pause an indexing job after its in-flight embedding request finishes, or resume it."
    )]
    async fn set_indexing_job_paused(
        &self,
        Parameters(args): Parameters<SetIndexJobPausedArgs>,
    ) -> Result<String, McpError> {
        let job_id = args.job_id.trim();
        if job_id.is_empty() || job_id.len() > 128 {
            return Err(McpError::invalid_params(
                "job_id must contain between 1 and 128 bytes",
                None,
            ));
        }
        self.index_jobs
            .set_paused(job_id, args.paused)
            .await
            .map_err(|error| McpError::invalid_request(error, None))?;
        serde_json::to_string(&serde_json::json!({
            "job_id": job_id,
            "paused": args.paused,
        }))
        .map_err(internal_error)
    }

    #[tool(
        description = "List the configured LFM embedding model and available llama-server reranking models."
    )]
    async fn list_models(&self) -> Result<String, McpError> {
        let endpoint =
            env::var("MODEL_LIST_URL").unwrap_or_else(|_| self.model_list_endpoint.clone());
        let mut request = reqwest::Client::new().get(endpoint);
        if let Ok(key) = env::var("EMBEDDING_API_KEY") {
            request = request.bearer_auth(key);
        }
        let response: ModelListResponse = request
            .send()
            .await
            .map_err(internal_error)?
            .error_for_status()
            .map_err(internal_error)?
            .json()
            .await
            .map_err(internal_error)?;

        let mut available = AvailableModelsResponse {
            embedding: vec![AvailableModel {
                id: self.embedding_model.clone(),
                tags: vec!["embedding".to_owned(), "llama-server".to_owned()],
            }],
            reranking: Vec::new(),
        };
        for model in response.data {
            let mut tags = model.tags;
            if let Some(status) = model.status {
                tags.extend(status.args);
            }
            if tags.iter().any(|tag| tag.contains("reranking")) {
                available
                    .reranking
                    .push(AvailableModel { id: model.id, tags });
            }
        }
        serde_json::to_string(&available).map_err(internal_error)
    }

    #[tool(
        description = "Embed 1 to 8 document passages with the configured LFM model through llama-server."
    )]
    async fn embed_texts(
        &self,
        Parameters(args): Parameters<EmbedTextsArgs>,
    ) -> Result<String, McpError> {
        if args.texts.is_empty() || args.texts.len() > 8 {
            return Err(McpError::invalid_params(
                "texts must contain between 1 and 8 passages",
                None,
            ));
        }
        if args
            .texts
            .iter()
            .any(|text| text.trim().is_empty() || text.len() > 16_384)
        {
            return Err(McpError::invalid_params(
                "each passage must be non-empty and at most 16384 bytes",
                None,
            ));
        }
        let model = args.model.as_deref().unwrap_or(&self.embedding_model);
        if model != self.embedding_model {
            return Err(McpError::invalid_params(
                "model must match the configured local embedding model",
                None,
            ));
        }
        let ctx = validate_embedding_ctx(args.ctx.unwrap_or(self.embedding_ctx))?;
        let embedder = self.embedder(model)?;
        let embeddings = embedder
            .embed(
                &args.texts,
                crate::test_pipeline::EmbeddingTask::Document,
                ctx,
            )
            .await
            .map_err(internal_error)?;
        if embeddings.len() != args.texts.len() {
            return Err(internal_error(
                "embedding service returned an unexpected number of vectors",
            ));
        }
        let dimensions = embeddings.first().map(Vec::len).unwrap_or(0);
        if dimensions == 0
            || embeddings.iter().any(|embedding| {
                embedding.len() != dimensions || embedding.iter().any(|value| !value.is_finite())
            })
        {
            return Err(internal_error(
                "embedding service returned vectors with invalid values or dimensions",
            ));
        }
        serde_json::to_string(&serde_json::json!({
            "model": embedder.model_name(),
            "count": embeddings.len(),
            "dimensions": dimensions,
        }))
        .map_err(internal_error)
    }

    #[tool(
        description = "Low-level embedding operation for up to 8 chunk IDs in a chunk table. Prefer start_indexing_job for resumable paper or corpus indexing."
    )]
    async fn embed_chunk_batch(
        &self,
        Parameters(args): Parameters<EmbedChunkBatchArgs>,
    ) -> Result<String, McpError> {
        if args.chunk_table.trim().is_empty() || args.chunk_table.len() > 128 {
            return Err(McpError::invalid_params(
                "chunk_table must contain between 1 and 128 bytes",
                None,
            ));
        }
        if args.chunk_ids.is_empty()
            || args.chunk_ids.len() > 8
            || args
                .chunk_ids
                .iter()
                .any(|id| id.is_empty() || id.len() > 256)
        {
            return Err(McpError::invalid_params(
                "chunk_ids must contain 1 to 8 non-empty chunk IDs",
                None,
            ));
        }
        if args.model.trim().is_empty() || args.model.len() > 512 {
            return Err(McpError::invalid_params(
                "model must contain between 1 and 512 bytes",
                None,
            ));
        }
        if args.model != self.embedding_model {
            return Err(McpError::invalid_params(
                "model must match the configured local embedding model",
                None,
            ));
        }
        let ctx = validate_embedding_ctx(args.ctx.unwrap_or(self.embedding_ctx))?;
        let table_name = args.chunk_table.trim();
        let database = connect(&self.database_path)
            .execute()
            .await
            .map_err(internal_error)?;
        let embedder = self.embedder(&args.model)?;
        let pipeline = crate::hybrid_pipeline::HybridPipeline::new(
            &database,
            crate::hybrid_pipeline::TableNames {
                chunks: table_name.to_owned(),
            },
        );
        let count = pipeline
            .embed_chunk_batch(&args.chunk_ids, &args.model, ctx, &embedder)
            .await
            .map_err(internal_error)?;
        serde_json::to_string(&serde_json::json!({
            "model": embedder.model_name(),
            "count": count,
            "chunk_table": table_name,
            "vector_column": crate::hybrid_pipeline::vector_column_name(&args.model),
        }))
        .map_err(internal_error)
    }

    #[tool(
        description = "Create the model-specific vector index after chunk embeddings have been persisted."
    )]
    async fn finalize_chunk_embeddings(
        &self,
        Parameters(args): Parameters<FinalizeChunkEmbeddingsArgs>,
    ) -> Result<String, McpError> {
        let table_name = args.chunk_table.trim();
        if table_name.is_empty() || table_name.len() > 128 {
            return Err(McpError::invalid_params(
                "chunk_table must contain between 1 and 128 bytes",
                None,
            ));
        }
        if args.model.trim().is_empty() || args.model.len() > 512 {
            return Err(McpError::invalid_params(
                "model must contain between 1 and 512 bytes",
                None,
            ));
        }
        let database = connect(&self.database_path)
            .execute()
            .await
            .map_err(internal_error)?;
        let pipeline = crate::hybrid_pipeline::HybridPipeline::new(
            &database,
            crate::hybrid_pipeline::TableNames {
                chunks: table_name.to_owned(),
            },
        );
        pipeline
            .ensure_vector_index(&args.model)
            .await
            .map_err(internal_error)?;
        serde_json::to_string(&serde_json::json!({
            "chunk_table": table_name,
            "vector_column": crate::hybrid_pipeline::vector_column_name(&args.model),
            "indexed": true,
        }))
        .map_err(internal_error)
    }

    #[tool(
        description = "Search chunks in a named LanceDB table using lexical and dense retrieval, reciprocal-rank fusion, and optional reranking."
    )]
    async fn search_hybrid_chunks(
        &self,
        Parameters(args): Parameters<HybridChunkSearchArgs>,
    ) -> Result<String, McpError> {
        let table_name = args.chunk_table.trim();
        let query = args.query.trim();
        let embedding_model = args.embedding_model.trim();
        if table_name.is_empty() || table_name.len() > 128 {
            return Err(McpError::invalid_params(
                "chunk_table must contain between 1 and 128 bytes",
                None,
            ));
        }
        if query.is_empty() || query.len() > 2_000 {
            return Err(McpError::invalid_params(
                "query must contain between 1 and 2000 bytes",
                None,
            ));
        }
        if embedding_model.is_empty() || embedding_model.len() > 512 {
            return Err(McpError::invalid_params(
                "embedding_model must contain between 1 and 512 bytes",
                None,
            ));
        }
        if args
            .reranker_model
            .as_deref()
            .is_some_and(|model| model.trim().is_empty() || model.len() > 512)
        {
            return Err(McpError::invalid_params(
                "reranker_model must be at most 512 bytes and must not be empty",
                None,
            ));
        }
        let candidate_limit = args.candidate_limit.unwrap_or(30);
        if candidate_limit == 0 || candidate_limit > 100 {
            return Err(McpError::invalid_params(
                "candidate_limit must be between 1 and 100",
                None,
            ));
        }
        let ctx = validate_embedding_ctx(args.ctx.unwrap_or(self.embedding_ctx))?;

        let database = connect(&self.database_path)
            .execute()
            .await
            .map_err(internal_error)?;
        let embedder = self.embedder(embedding_model)?;
        let reranker = args
            .reranker_model
            .as_deref()
            .map(|model| self.reranker(model))
            .transpose()?;
        let pipeline = crate::hybrid_pipeline::HybridPipeline::new(
            &database,
            crate::hybrid_pipeline::TableNames {
                chunks: table_name.to_owned(),
            },
        );
        let results = pipeline
            .search(
                query,
                embedding_model,
                reranker
                    .as_ref()
                    .map(|reranker| reranker as &dyn crate::test_pipeline::Reranker),
                candidate_limit,
                ctx,
                &embedder,
            )
            .await
            .map_err(internal_error)?;
        serde_json::to_string(&results).map_err(internal_error)
    }

    #[tool(
        description = "Index or resume a paper through the shared corpus pipeline, then test it with lexical, dense, reciprocal-rank-fusion, and optional reranking."
    )]
    async fn test_paper_chunks(
        &self,
        Parameters(args): Parameters<TestPaperArgs>,
    ) -> Result<String, McpError> {
        let paper_id = args.paper_id.trim();
        if paper_id.is_empty() || paper_id.len() > 128 {
            return Err(McpError::invalid_params(
                "paper_id must contain between 1 and 128 bytes",
                None,
            ));
        }
        let query = args.query.trim();
        if query.is_empty() || query.len() > 2_000 {
            return Err(McpError::invalid_params(
                "query must contain between 1 and 2000 bytes",
                None,
            ));
        }
        if args
            .embedding_model
            .as_deref()
            .is_some_and(|model| model.trim().is_empty() || model.len() > 512)
        {
            return Err(McpError::invalid_params(
                "embedding_model must be at most 512 bytes and must not be empty",
                None,
            ));
        }
        if args
            .reranker_model
            .as_deref()
            .is_some_and(|model| model.trim().is_empty() || model.len() > 512)
        {
            return Err(McpError::invalid_params(
                "reranker_model must be at most 512 bytes and must not be empty",
                None,
            ));
        }
        let candidate_limit = args.candidate_limit.unwrap_or(30);
        if candidate_limit == 0 || candidate_limit > 100 {
            return Err(McpError::invalid_params(
                "candidate_limit must be between 1 and 100",
                None,
            ));
        }
        let ctx = validate_embedding_ctx(args.ctx.unwrap_or(self.embedding_ctx))?;
        let table_name = crate::hybrid_pipeline::GLOBAL_CHUNK_TABLE;
        let database = connect(&self.database_path)
            .execute()
            .await
            .map_err(internal_error)?;
        let embedding_model = args
            .embedding_model
            .as_deref()
            .unwrap_or(&self.embedding_model);
        if embedding_model != self.embedding_model {
            return Err(McpError::invalid_params(
                "embedding_model must match the configured local embedding model",
                None,
            ));
        }
        self.index_one_paper(paper_id, ctx, DEFAULT_CHUNK_TOKEN_BUDGET)
            .await
            .map_err(internal_error)?;
        let embedder = self.embedder(embedding_model)?;
        let reranker = args
            .reranker_model
            .as_deref()
            .map(|model| self.reranker(model))
            .transpose()?;
        let pipeline = crate::hybrid_pipeline::HybridPipeline::new(
            &database,
            crate::hybrid_pipeline::TableNames {
                chunks: table_name.to_owned(),
            },
        );
        let result = pipeline
            .search(
                query,
                embedder.model_name(),
                reranker
                    .as_ref()
                    .map(|reranker| reranker as &dyn crate::test_pipeline::Reranker),
                candidate_limit,
                ctx,
                &embedder,
            )
            .await
            .map_err(internal_error)?;
        serde_json::to_string_pretty(&serde_json::json!({
            "paper_id": paper_id,
            "result": result,
        }))
        .map_err(internal_error)
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
    let config: AppConfig = serde_json::from_slice(&std::fs::read("config.json")?)?;
    if config.models.embedding.id.trim().is_empty()
        || config.models.embedding.repo.trim().is_empty()
        || config.models.embedding.path.trim().is_empty()
        || config.models.embedding.tokenizer_path.trim().is_empty()
        || !(crate::test_pipeline::EMBEDDING_TOKEN_HEADROOM + 2..=MAX_LFM_EMBEDDING_TOKENS)
            .contains(&config.models.embedding.max_tokens)
    {
        return Err(format!(
            "models.embedding requires non-empty id, repo, path, and tokenizerPath; maxTokens must be from {} to {MAX_LFM_EMBEDDING_TOKENS}",
            crate::test_pipeline::EMBEDDING_TOKEN_HEADROOM + 2
        )
        .into());
    }
    let tokenizer_path = &config.models.embedding.tokenizer_path;
    let tokenizer = Arc::new(Tokenizer::from_file(tokenizer_path).map_err(|error| {
        let local_dir = std::path::Path::new(tokenizer_path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        std::io::Error::other(format!(
            "failed to load tokenizer at {tokenizer_path}: {error}. Download it manually with: hf download {} tokenizer.json --local-dir '{}'",
            config.models.embedding.repo,
            local_dir.display()
        ))
    })?);
    let api_url = reqwest::Url::parse(&config.api_base_url)?;
    if !matches!(api_url.scheme(), "http" | "https") {
        return Err("apiBaseUrl must use http or https".into());
    }
    let bind_address = SocketAddr::new(config.backend.host, config.backend.port);
    let database_path =
        env::var("ARXIV_LANCE_DB").unwrap_or_else(|_| DEFAULT_DATABASE_PATH.to_owned());
    let table_name =
        env::var("ARXIV_LANCE_TABLE").unwrap_or_else(|_| DEFAULT_TABLE_NAME.to_owned());

    let database = connect(&database_path).execute().await?;
    let table = database.open_table(&table_name).execute().await?;
    ensure_indexing_state_fields(&table).await?;
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
    let index_jobs = IndexJobManager::new();
    let server = ArxivMcpServer::new(
        table,
        metadata_table,
        table_name,
        row_count,
        metadata_row_count,
        database_path.clone(),
        config.models.embedding.clone(),
        format!(
            "http://{}/v1/embeddings",
            SocketAddr::new(config.llama.host, config.llama.port)
        ),
        format!(
            "http://{}/v1/models",
            SocketAddr::new(config.llama.host, config.llama.port)
        ),
        format!(
            "http://{}/rerank",
            SocketAddr::new(config.llama.host, config.llama.port)
        ),
        tokenizer,
        Arc::new(Semaphore::new(1)),
        index_jobs.clone(),
    );

    let service: StreamableHttpService<ArxivMcpServer, LocalSessionManager> =
        StreamableHttpService::new(
            move || Ok(server.clone()),
            Arc::new(LocalSessionManager::default()),
            Default::default(),
        );
    let runtime_config = config.clone();
    let event_jobs = index_jobs.clone();
    let router = Router::new()
        .route("/health", axum::routing::get(|| async { "ok" }))
        .route(
            "/indexing-jobs/{job_id}/events",
            get(move |Path(job_id): Path<String>| {
                let manager = event_jobs.clone();
                async move { index_job_events(manager, job_id).await }
            }),
        )
        .route(
            "/runtime-config.json",
            axum::routing::get(move || {
                let config = runtime_config.clone();
                async move { Json(config) }
            }),
        )
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

async fn ensure_indexing_state_fields(table: &Table) -> Result<(), Box<dyn Error>> {
    let schema = table.schema().await?;
    let expressions = [
        ("chunked", "false"),
        ("embedded", "false"),
        ("embedding_model", "CAST(NULL AS STRING)"),
        ("chunk_count", "CAST(NULL AS INTEGER)"),
        ("embedded_chunk_count", "CAST(NULL AS INTEGER)"),
        ("indexing_error", "CAST(NULL AS STRING)"),
    ];
    let missing_expressions = expressions
        .into_iter()
        .filter(|(name, _)| schema.field_with_name(name).is_err())
        .map(|(name, expression)| (name.to_owned(), expression.to_owned()))
        .collect::<Vec<_>>();
    if !missing_expressions.is_empty() {
        table
            .add_columns()
            .transform(NewColumnTransform::SqlExpressions(missing_expressions))
            .execute()
            .await?;
    }

    let indices = table.list_indices().await?;
    if !indices.iter().any(|index| {
        index.index_type.to_string() == "BTREE"
            && index.columns.iter().any(|column| column == "paper_id")
    }) {
        table
            .create_index(&["paper_id"], Index::BTree(BTreeIndexBuilder::default()))
            .execute()
            .await?;
    }
    for column in ["chunked", "embedded"] {
        if !indices.iter().any(|index| {
            index.index_type.to_string().eq_ignore_ascii_case("bitmap")
                && index.columns.iter().any(|indexed| indexed == column)
        }) {
            table
                .create_index(&[column], Index::Bitmap(BitmapIndexBuilder::default()))
                .execute()
                .await?;
        }
    }
    Ok(())
}

fn sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn pending_index_filter(model: &str) -> String {
    format!(
        "chunked IS NULL OR chunked = false OR embedded IS NULL OR embedded = false OR embedding_model IS NULL OR embedding_model != {}",
        sql_string(model)
    )
}

async fn update_paper_index_state(
    table: &Table,
    paper_id: &str,
    fields: &[(&str, String)],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut update = table
        .update()
        .only_if(format!("paper_id = {}", sql_string(paper_id)));
    for (field, expression) in fields {
        update = update.column(*field, expression.clone());
    }
    let result = update.execute().await?;
    if result.rows_updated != 1 {
        return Err(format!(
            "expected to update one indexing-state row for paper {paper_id:?}, updated {}",
            result.rows_updated
        )
        .into());
    }
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

fn string_at(
    batch: &lancedb::arrow::arrow_array::RecordBatch,
    column: &str,
    row: usize,
) -> Result<Option<String>, Box<dyn Error + Send + Sync>> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| std::io::Error::other(format!("missing column {column:?}")))?;
    string_value(array.as_ref(), row)
        .map(|value| value.map(str::to_owned))
        .map_err(std::io::Error::other)
        .map_err(Into::into)
}

fn optional_bool_value(
    array: &dyn Array,
    row: usize,
) -> Result<Option<bool>, Box<dyn Error + Send + Sync>> {
    if array.is_null(row) {
        return Ok(None);
    }
    let values = array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| std::io::Error::other("indexing state column must be Boolean"))?;
    Ok(Some(values.value(row)))
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

async fn index_job_events(
    manager: IndexJobManager,
    job_id: String,
) -> Result<
    Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>,
    axum::http::StatusCode,
> {
    let (initial, receiver) = manager
        .subscribe(&job_id)
        .await
        .map_err(|_| axum::http::StatusCode::NOT_FOUND)?;
    let updates = stream::unfold(
        (Some(initial), receiver, job_id, false),
        |(initial, mut receiver, job_id, finished)| async move {
            if finished {
                return None;
            }
            let progress = if let Some(initial) = initial {
                initial
            } else {
                loop {
                    match receiver.recv().await {
                        Ok(progress) if progress.job_id == job_id => break progress,
                        Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(broadcast::error::RecvError::Closed) => return None,
                    }
                }
            };
            let finished = matches!(progress.state.as_str(), "complete" | "failed");
            let data = serde_json::to_string(&progress)
                .expect("serializing indexing progress cannot fail");
            Some((
                Ok(Event::default().event("progress").data(data)),
                (None, receiver, job_id, finished),
            ))
        },
    );
    Ok(Sse::new(updates).keep_alive(KeepAlive::default()))
}

fn validate_embedding_ctx(ctx: usize) -> Result<usize, McpError> {
    if (1..=DEFAULT_EMBEDDING_CTX).contains(&ctx) {
        Ok(ctx)
    } else {
        Err(McpError::invalid_params(
            format!("ctx must be between 1 and {DEFAULT_EMBEDDING_CTX}"),
            None,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_EMBEDDING_CTX, IndexJobManager, MAX_LFM_EMBEDDING_TOKENS, SEARCH_SNIPPET_CHARS,
        ensure_indexing_state_fields, pending_index_filter, search_excerpt, sql_string,
        update_paper_index_state, validate_embedding_ctx,
    };
    use lancedb::{
        arrow::{
            arrow_array::{RecordBatch, StringArray},
            arrow_schema::{DataType, Field, Schema},
        },
        connect,
    };
    use std::sync::Arc;

    #[test]
    fn embedding_ctx_accepts_limits_and_rejects_out_of_range_values() {
        assert_eq!(validate_embedding_ctx(1).unwrap(), 1);
        assert_eq!(
            validate_embedding_ctx(DEFAULT_EMBEDDING_CTX).unwrap(),
            DEFAULT_EMBEDDING_CTX
        );
        assert!(validate_embedding_ctx(0).is_err());
        assert!(validate_embedding_ctx(DEFAULT_EMBEDDING_CTX + 1).is_err());
    }

    #[test]
    fn embedding_limit_reserves_headroom_below_model_context() {
        assert_eq!(
            crate::test_pipeline::effective_embedding_token_limit(
                DEFAULT_EMBEDDING_CTX,
                MAX_LFM_EMBEDDING_TOKENS
            )
            .unwrap(),
            MAX_LFM_EMBEDDING_TOKENS - crate::test_pipeline::EMBEDDING_TOKEN_HEADROOM
        );
        assert!(
            crate::test_pipeline::effective_embedding_token_limit(
                crate::test_pipeline::EMBEDDING_TOKEN_HEADROOM + 1,
                MAX_LFM_EMBEDDING_TOKENS
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn indexing_job_pause_waits_for_resume() {
        let jobs = IndexJobManager::new();
        let job_id = jobs.start(None, 1).await.unwrap();
        jobs.update(|progress| progress.state = "running".to_owned())
            .await;
        jobs.set_paused(&job_id, true).await.unwrap();

        let waiter_jobs = jobs.clone();
        let mut waiter = tokio::spawn(async move {
            waiter_jobs.wait_until_resumed().await;
        });
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), &mut waiter)
                .await
                .is_err()
        );

        jobs.set_paused(&job_id, false).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn indexing_state_flags_are_filterable_and_updateable() {
        let database = connect("memory://").execute().await.unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("paper_id", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["paper-1"])),
                Arc::new(StringArray::from(vec!["paper text"])),
            ],
        )
        .unwrap();
        let table = database
            .create_table("paper_state_test", batch)
            .execute()
            .await
            .unwrap();

        ensure_indexing_state_fields(&table).await.unwrap();
        update_paper_index_state(
            &table,
            "paper-1",
            &[
                ("chunked", "true".to_owned()),
                ("embedded", "true".to_owned()),
                ("embedding_model", sql_string("model-1")),
                ("chunk_count", "3".to_owned()),
                ("embedded_chunk_count", "3".to_owned()),
                ("indexing_error", "NULL".to_owned()),
            ],
        )
        .await
        .unwrap();

        assert_eq!(
            table
                .count_rows(Some("chunked = true".to_owned()))
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            table
                .count_rows(Some(pending_index_filter("model-1")))
                .await
                .unwrap(),
            0
        );
    }

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
