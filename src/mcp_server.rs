use crate::python_indexer::{IndexBuildProgress, PythonIndexClient};
use crate::test_pipeline::Embedder;
use axum::{
    Json, Router,
    extract::Path,
    http::{HeaderName, StatusCode},
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
    collections::{BTreeSet, HashMap, VecDeque},
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
    total_papers_known: bool,
    phase_completed: usize,
    phase_total: Option<usize>,
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
    #[serde(skip_serializing)]
    embedding_completions: VecDeque<(Instant, usize)>,
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
            total_papers_known: total_papers > 0,
            phase_completed: 0,
            phase_total: None,
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
            embedding_completions: VecDeque::new(),
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
            let now = Instant::now();
            while progress
                .embedding_completions
                .front()
                .is_some_and(|(completed_at, _)| {
                    now.duration_since(*completed_at) > Duration::from_secs(30)
                })
            {
                progress.embedding_completions.pop_front();
            }
            let recent_completed: usize = progress
                .embedding_completions
                .iter()
                .map(|(_, count)| count)
                .sum();
            progress.papers_per_second = progress
                .embedding_completions
                .front()
                .map(|(first, _)| {
                    let elapsed = now.duration_since(*first).as_secs_f64().max(1.0);
                    recent_completed as f64 / elapsed
                })
                .unwrap_or(0.0);
            progress.eta_seconds = if progress.total_papers_known
                && progress.scanned_papers < progress.total_papers
                && progress.papers_per_second > 0.0
                && progress.state == "running"
            {
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

    async fn record_embedding_completion(&self, papers: usize) {
        if papers == 0 {
            return;
        }
        if let Some(progress) = self.progress.write().await.as_mut() {
            progress
                .embedding_completions
                .push_back((Instant::now(), papers));
        }
    }

    async fn finish(&self, state: &str, error: Option<String>) {
        self.update(|progress| {
            progress.state = if state == "complete" && progress.failed_papers > 0 {
                "complete_with_errors"
            } else {
                state
            }
            .to_owned();
            progress.current_paper = None;
            progress.phase = Some(
                if state == "complete" && progress.total_papers == 0 {
                    "No papers pending indexing"
                } else {
                    state
                }
                .to_owned(),
            );
            if error.is_some() || progress.failed_papers == 0 {
                progress.error = error;
            }
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
        if progress.phase.as_deref().is_some_and(|phase| {
            phase.starts_with("Building full-text vector index")
                || phase.starts_with("Building abstract vector index")
                || phase.starts_with("GPU IVF-PQ:")
        }) {
            return Err("IVF-PQ index builds cannot be paused".to_owned());
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

    async fn current(&self) -> Option<IndexJobProgress> {
        self.progress
            .read()
            .await
            .clone()
            .filter(|progress| matches!(progress.state.as_str(), "queued" | "running" | "paused"))
    }
}

#[derive(Clone, Default)]
struct IndexBuildProgressHub {
    updates: Arc<RwLock<HashMap<String, broadcast::Sender<IndexBuildProgress>>>>,
}

impl IndexBuildProgressHub {
    async fn subscribe(
        &self,
        job_id: &str,
    ) -> Result<broadcast::Receiver<IndexBuildProgress>, String> {
        let mut updates = self.updates.write().await;
        if updates.contains_key(job_id) {
            return Err(format!("Index build job {job_id:?} is already registered"));
        }
        let (sender, receiver) = broadcast::channel(64);
        updates.insert(job_id.to_owned(), sender);
        Ok(receiver)
    }

    async fn publish(&self, update: IndexBuildProgress) -> Result<(), String> {
        let updates = self.updates.read().await;
        let sender = updates
            .get(&update.job_id)
            .ok_or_else(|| format!("No active index build for {:?}", update.job_id))?;
        sender
            .send(update)
            .map(|_| ())
            .map_err(|error| format!("Index build progress receiver unavailable: {error}"))
    }

    async fn remove(&self, job_id: &str) {
        self.updates.write().await.remove(job_id);
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
            let vectors = self.inner.embed(texts, task, ctx).await?;
            if let Some(jobs) = self.jobs {
                jobs.record_embedding_completion(texts.len()).await;
            }
            Ok(vectors)
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
struct EmbeddedSearchArgs {
    #[schemars(
        description = "Optional search text; empty query browses papers with vectors for the selected embedding model"
    )]
    query: Option<String>,
    #[schemars(
        description = "Embedding model whose non-null vectors define the searchable subset"
    )]
    embedding_model: Option<String>,
    #[schemars(description = "Maximum results to return, from 1 to 10")]
    limit: Option<usize>,
    #[schemars(description = "Number of results to skip")]
    offset: Option<usize>,
    #[schemars(
        description = "Use semantic-only abstract-vector search instead of hybrid lexical and semantic search"
    )]
    semantic_only: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AbstractIndexArgs {
    #[schemars(description = "Embedding model whose abstract vector index is checked or built")]
    embedding_model: Option<String>,
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
    #[schemars(
        description = "Embedding mode: abstract (default) or full_text; abstract dense results are fused with full-text lexical results by paper_id"
    )]
    mode: Option<String>,
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
    #[schemars(description = "Embedding mode: abstract (default) or full_text")]
    mode: Option<String>,
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
    python_indexer: PythonIndexClient,
    model_list_endpoint: String,
    reranker_endpoint: String,
    embedding_tokenizer: Arc<Tokenizer>,
    embedding_scheduler: Arc<Semaphore>,
    embedding_ctx: usize,
    embedding_max_tokens: usize,
    index_jobs: IndexJobManager,
    index_build_progress: IndexBuildProgressHub,
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
        python_indexer_endpoint: String,
        python_indexer_callback_endpoint: String,
        model_list_endpoint: String,
        reranker_endpoint: String,
        embedding_tokenizer: Arc<Tokenizer>,
        embedding_scheduler: Arc<Semaphore>,
        index_jobs: IndexJobManager,
        index_build_progress: IndexBuildProgressHub,
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
            python_indexer: PythonIndexClient::new(
                python_indexer_endpoint,
                python_indexer_callback_endpoint,
            ),
            model_list_endpoint,
            reranker_endpoint,
            embedding_tokenizer,
            embedding_scheduler,
            embedding_ctx: DEFAULT_EMBEDDING_CTX,
            embedding_max_tokens: embedding.max_tokens,
            index_jobs,
            index_build_progress,
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

    async fn open_fresh_metadata_table(&self) -> Result<Table, Box<dyn Error + Send + Sync>> {
        let table_name =
            env::var("ARXIV_METADATA_TABLE").unwrap_or_else(|_| "arxiv_metadata".to_owned());
        let database = connect(&self.database_path).execute().await?;
        Ok(database.open_table(&table_name).execute().await?)
    }

    async fn build_gpu_vector_index(
        &self,
        table_name: &str,
        column_name: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let job_id = self
            .index_jobs
            .current()
            .await
            .map(|job| job.job_id)
            .unwrap_or_else(|| {
                format!(
                    "index-{}",
                    NEXT_INDEX_JOB_ID.fetch_add(1, Ordering::Relaxed)
                )
            });
        let mut progress = self
            .index_build_progress
            .subscribe(&job_id)
            .await
            .map_err(std::io::Error::other)?;
        let result = async {
            self.python_indexer
                .start_index(&job_id, &self.database_path, table_name, column_name)
                .await?;
            let mut last_stage = String::new();
            loop {
                let update = progress.recv().await.map_err(|error| {
                    std::io::Error::other(format!("Python index progress stream failed: {error}"))
                })?;
                let state = update.state.clone();
                if update.stage != last_stage {
                    eprintln!(
                        "[gpu-vector-index] progress state={} job_id={} stage={:?} completed={:?} total={:?}",
                        update.state,
                        update.job_id,
                        update.stage,
                        update.completed,
                        update.total
                    );
                    last_stage.clone_from(&update.stage);
                }
                self.update_index_build_progress(&job_id, update.clone())
                    .await;
                match state.as_str() {
                    "completed" => {
                        eprintln!("[gpu-vector-index] Python worker finished job_id={job_id}");
                        return Ok(());
                    }
                    "failed" => {
                        return Err(format!(
                            "Python IVF-PQ build failed: {}",
                            update.error.as_deref().unwrap_or("unknown error")
                        )
                        .into());
                    }
                    "queued" | "running" => {}
                    unexpected => {
                        return Err(format!(
                            "Python indexer returned unexpected job state {unexpected:?}"
                        )
                        .into());
                    }
                }
            }
        }
        .await;
        self.index_build_progress.remove(&job_id).await;
        result
    }

    async fn update_index_build_progress(&self, job_id: &str, update: IndexBuildProgress) {
        if matches!(update.state.as_str(), "completed" | "failed") {
            eprintln!(
                "[gpu-vector-index] Python worker state={} job_id={} stage={:?} completed={:?} total={:?} error={:?}",
                update.state,
                update.job_id,
                update.stage,
                update.completed,
                update.total,
                update.error
            );
        }
        if self
            .index_jobs
            .current()
            .await
            .is_some_and(|job| job.job_id == job_id)
        {
            self.index_jobs
                .update(|progress| {
                    progress.phase = Some(format!("GPU IVF-PQ: {}", update.stage));
                    progress.phase_completed = update.completed.unwrap_or_default();
                    progress.phase_total = update.total;
                    if update.state == "failed" {
                        progress.error = update.error;
                    }
                })
                .await;
        }
    }

    async fn ensure_gpu_vector_index(
        &self,
        table_name: &str,
        embedding_model: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let column_name = crate::hybrid_pipeline::vector_column_name(embedding_model);
        let database = connect(&self.database_path).execute().await?;
        let table = database.open_table(table_name).execute().await?;
        let schema = table.schema().await?;
        schema.field_with_name(&column_name)?;
        let vector_count = table
            .count_rows(Some(format!("{column_name} IS NOT NULL")))
            .await?;
        if vector_count == 0 {
            return Err(format!("no embeddings found in vector column {column_name:?}").into());
        }
        let existing_index = table.list_indices().await?.into_iter().find(|index| {
            index.index_type.to_string().eq_ignore_ascii_case("IVF_PQ")
                && index.columns.iter().any(|column| column == &column_name)
        });
        let index_is_current = if let Some(index) = existing_index {
            let stats = table.index_stats(&index.name).await?;
            if let Some(stats) = &stats {
                eprintln!(
                    "[gpu-vector-index] existing index stats name={:?} covered_table_rows={} unindexed_table_rows={}",
                    index.name, stats.num_indexed_rows, stats.num_unindexed_rows
                );
            }
            stats.is_some_and(|stats| {
                stats.num_indexed_rows >= vector_count && stats.num_unindexed_rows == 0
            })
        } else {
            false
        };
        if index_is_current {
            return Ok(());
        }

        self.rebuild_gpu_vector_index(table_name, embedding_model)
            .await
    }

    async fn rebuild_gpu_vector_index(
        &self,
        table_name: &str,
        embedding_model: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let column_name = crate::hybrid_pipeline::vector_column_name(embedding_model);
        self.build_gpu_vector_index(table_name, &column_name)
            .await?;
        let refreshed_database = connect(&self.database_path).execute().await?;
        let refreshed_table = refreshed_database.open_table(table_name).execute().await?;
        let vector_count = refreshed_table
            .count_rows(Some(format!("{column_name} IS NOT NULL")))
            .await?;
        let rebuilt_index = refreshed_table
            .list_indices()
            .await?
            .into_iter()
            .find(|index| {
                index.index_type.to_string().eq_ignore_ascii_case("IVF_PQ")
                    && index.columns.iter().any(|column| column == &column_name)
            });
        let index_is_current = if let Some(index) = rebuilt_index {
            let stats = refreshed_table.index_stats(&index.name).await?;
            if let Some(stats) = &stats {
                eprintln!(
                    "[gpu-vector-index] rebuilt index stats name={:?} covered_table_rows={} unindexed_table_rows={} vector_count={vector_count}",
                    index.name, stats.num_indexed_rows, stats.num_unindexed_rows
                );
            }
            stats.is_some_and(|stats| {
                stats.num_indexed_rows >= vector_count && stats.num_unindexed_rows == 0
            })
        } else {
            false
        };
        if !index_is_current {
            return Err(format!(
                "Python indexer completed without covering all {vector_count} vectors in {column_name:?}"
            )
            .into());
        }
        Ok(())
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
        mode: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.index_jobs
            .update(|progress| {
                progress.phase = Some("Connecting to database".to_owned());
            })
            .await;
        let database = connect(&self.database_path).execute().await?;
        if mode == "abstract" {
            return self.run_abstract_index_job(&database, paper_id).await;
        }
        let filter = if let Some(paper_id) = &paper_id {
            format!("paper_id = {}", sql_string(paper_id))
        } else {
            pending_index_filter(&self.embedding_model)
        };
        self.index_jobs
            .update(|progress| {
                progress.phase = Some("Counting pending papers".to_owned());
            })
            .await;
        let total = self.table.count_rows(Some(filter.clone())).await?;
        self.index_jobs
            .update(|progress| {
                progress.total_papers = total;
                progress.total_papers_known = true;
                progress.phase = Some("Reading pending papers".to_owned());
            })
            .await;
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
                        self.index_jobs
                            .update(|progress| {
                                progress.error = Some(format!("{current_id}: {error}"));
                            })
                            .await;
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
            self.index_jobs
                .update(|progress| {
                    progress.phase = Some("Building full-text vector index".to_owned());
                    progress.current_paper = None;
                })
                .await;
            self.index_jobs.wait_until_resumed().await;
            self.ensure_gpu_vector_index(
                crate::hybrid_pipeline::GLOBAL_CHUNK_TABLE,
                &self.embedding_model,
            )
            .await?;
        }
        Ok(())
    }

    async fn run_abstract_index_job(
        &self,
        _database: &lancedb::Connection,
        paper_id: Option<String>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let table = &self.metadata_table;
        self.index_jobs
            .update(|progress| {
                progress.phase = Some("Counting pending abstracts".to_owned());
                progress.phase_completed = 0;
                progress.phase_total = None;
            })
            .await;
        let ready_column =
            crate::hybrid_pipeline::embedding_ready_column_name(&self.embedding_model);
        let has_ready_column = table.schema().await?.field_with_name(&ready_column).is_ok();
        let mut filters = Vec::new();
        if let Some(id) = &paper_id {
            filters.push(format!("paper_id = {}", sql_string(id)));
        } else if has_ready_column {
            filters.push(format!("{ready_column} IS NULL OR {ready_column} = false"));
        }
        let filter = (!filters.is_empty()).then(|| filters.join(" AND "));
        let total = table.count_rows(filter.clone()).await?;
        self.index_jobs
            .update(|progress| {
                progress.total_papers = total;
                progress.total_papers_known = true;
                progress.phase = Some("Reading pending abstracts".to_owned());
            })
            .await;
        let mut query = table.query();
        if let Some(filter) = filter {
            query = query.only_if(filter);
        }
        let mut rows = query
            .select(Select::columns(&["paper_id", "abstract"]))
            .execute()
            .await?;
        let embedder = self.embedder(&self.embedding_model)?;
        let job_embedder = JobPausableEmbedder {
            inner: &embedder,
            jobs: Some(&self.index_jobs),
        };
        let mut scanned = 0;
        let mut embedded_count = 0;
        let mut failed_count = 0;
        let mut last_progress_update = Instant::now();
        const EMBEDDING_BATCH_SIZE: usize = 32;
        while let Some(batch) = rows.next().await {
            let batch = batch?;
            let mut papers = Vec::new();
            for row in 0..batch.num_rows() {
                let id = string_at(&batch, "paper_id", row)?
                    .ok_or("abstract row has a null paper_id")?;
                let abstract_text = string_at(&batch, "abstract", row)?.unwrap_or_default();
                papers.push((id, abstract_text));
            }
            for group in papers.chunks(EMBEDDING_BATCH_SIZE) {
                self.index_jobs.wait_until_resumed().await;
                let current = group.last().map(|(id, _)| id.clone());
                self.index_jobs
                    .update(|progress| progress.phase = Some("embedding abstracts".to_owned()))
                    .await;
                match crate::hybrid_pipeline::embed_abstract_batch(
                    table,
                    group,
                    &self.embedding_model,
                    self.embedding_ctx,
                    &job_embedder,
                )
                .await
                {
                    Ok(count) => embedded_count += count,
                    Err(error) => {
                        failed_count += group.len();
                        eprintln!("Abstract batch embedding failed: {error}");
                        self.index_jobs
                            .update(|progress| progress.error = Some(error.to_string()))
                            .await;
                    }
                }
                scanned += group.len();
                if scanned % 100 < EMBEDDING_BATCH_SIZE
                    || paper_id.is_some()
                    || last_progress_update.elapsed().as_secs_f64() >= 1.0
                {
                    self.index_jobs
                        .update(|progress| {
                            progress.scanned_papers = scanned;
                            progress.chunked_papers = 0;
                            progress.embedded_papers = embedded_count;
                            progress.failed_papers = failed_count;
                            progress.current_paper = current;
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
                progress.chunked_papers = 0;
                progress.embedded_papers = embedded_count;
                progress.failed_papers = failed_count;
            })
            .await;
        if embedded_count > 0 {
            self.index_jobs
                .update(|progress| {
                    progress.phase = Some("Building abstract vector index".to_owned());
                })
                .await;
            self.index_jobs.wait_until_resumed().await;
            let table_name =
                env::var("ARXIV_METADATA_TABLE").unwrap_or_else(|_| "arxiv_metadata".to_owned());
            self.ensure_gpu_vector_index(&table_name, &self.embedding_model)
                .await?;
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
            let result = self
                .process_paper(
                    &database,
                    &id,
                    &text,
                    chunked,
                    embedded,
                    model.as_deref(),
                    ctx,
                    token_budget,
                    false,
                )
                .await?;
            if result.1 {
                self.ensure_gpu_vector_index(
                    crate::hybrid_pipeline::GLOBAL_CHUNK_TABLE,
                    &self.embedding_model,
                )
                .await?;
            }
            return Ok(result);
        }
        Err(format!("No paper found with paper_id {paper_id:?}").into())
    }

    async fn process_paper(
        &self,
        database: &lancedb::Connection,
        paper_id: &str,
        text: &str,
        mut chunked: Option<bool>,
        mut embedded: Option<bool>,
        indexed_model: Option<&str>,
        ctx: usize,
        token_budget: usize,
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
        if chunked == Some(true) && !(embedded == Some(true) && same_model) {
            let safe_limit = crate::test_pipeline::effective_embedding_token_limit(
                ctx,
                self.embedding_max_tokens,
            )?;
            let stored_chunks = pipeline.paper_chunk_texts(paper_id).await?;
            let mut must_rechunk = stored_chunks.is_empty();
            for text in &stored_chunks {
                let input = format!("document: {text}");
                if self
                    .embedding_tokenizer
                    .encode(input, true)?
                    .get_ids()
                    .len()
                    > safe_limit
                {
                    must_rechunk = true;
                    break;
                }
            }
            if must_rechunk {
                pipeline.delete_paper_chunks(paper_id).await?;
                update_paper_index_state(
                    &self.table,
                    paper_id,
                    &[
                        ("chunked", "false".to_owned()),
                        ("embedded", "false".to_owned()),
                        ("chunk_count", "0".to_owned()),
                        ("embedded_chunk_count", "0".to_owned()),
                        (
                            "indexing_error",
                            sql_string("Stored chunks exceeded the safe embedding limit; regenerating them with the configured tokenizer."),
                        ),
                    ],
                )
                .await?;
                chunked = Some(false);
                embedded = Some(false);
            }
        }
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
        let embedding = pipeline
            .embed_paper_chunks_without_index(
                paper_id,
                &self.embedding_model,
                ctx,
                &job_embedder,
                4,
            )
            .await;
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

    #[tool(
        description = "Browse papers with abstract embeddings, run hybrid lexical+semantic search, or use semantic-only search for a model. An empty query browses the embedded subset."
    )]
    async fn search_embedded_papers(
        &self,
        Parameters(args): Parameters<EmbeddedSearchArgs>,
    ) -> Result<String, McpError> {
        let started_at = Instant::now();
        let query = args.query.as_deref().unwrap_or("").trim();
        if query.len() > 2_000 {
            return Err(McpError::invalid_params(
                "query must be at most 2000 bytes",
                None,
            ));
        }
        let limit = args.limit.unwrap_or(10).min(MAX_SEARCH_RESULTS);
        if limit == 0 {
            return Err(McpError::invalid_params("limit must be at least 1", None));
        }
        let offset = args.offset.unwrap_or(0);
        let model = args
            .embedding_model
            .as_deref()
            .unwrap_or(&self.embedding_model);
        let semantic_only = args.semantic_only.unwrap_or(false);
        if model.trim().is_empty() || model.len() > 512 {
            return Err(McpError::invalid_params("embedding_model is invalid", None));
        }
        let ready_column = crate::hybrid_pipeline::embedding_ready_column_name(model);
        eprintln!(
            "[embedded-search] start mode={} model={model:?} query_chars={} offset={offset} limit={limit}",
            if query.is_empty() {
                "browse"
            } else if semantic_only {
                "semantic"
            } else {
                "hybrid"
            },
            query.chars().count()
        );
        self.metadata_table
            .schema()
            .await
            .map_err(internal_error)?
            .field_with_name(&ready_column)
            .map_err(internal_error)?;

        let (results, has_more) = if query.is_empty() {
            let browse_started = Instant::now();
            eprintln!("[embedded-search] browse query started readiness_column={ready_column}");
            let mut rows = self
                .metadata_table
                .query()
                .only_if(format!("{ready_column} = true"))
                .select(Select::columns(&["paper_id", "abstract"]))
                .limit(limit.saturating_add(1))
                .offset(offset)
                .execute()
                .await
                .map_err(internal_error)?;
            let mut hits = Vec::new();
            let mut scanned_rows = 0usize;
            while let Some(batch) = rows.next().await {
                let batch = batch.map_err(internal_error)?;
                scanned_rows += batch.num_rows();
                for row in 0..batch.num_rows() {
                    let paper_id = string_at(&batch, "paper_id", row)
                        .map_err(internal_error)?
                        .ok_or_else(|| internal_error("embedded metadata row has null paper_id"))?;
                    let snippet = string_at(&batch, "abstract", row)
                        .map_err(internal_error)?
                        .unwrap_or_default()
                        .to_owned();
                    let metadata = self.metadata_for_id(&paper_id).await?;
                    hits.push(SearchHit {
                        paper_id,
                        metadata,
                        snippet,
                        snippet_truncated: false,
                    });
                }
            }
            let has_more = hits.len() > limit;
            hits.truncate(limit);
            eprintln!(
                "[embedded-search] browse query complete elapsed_ms={} rows={} returned={} has_more={has_more}",
                browse_started.elapsed().as_millis(),
                scanned_rows,
                hits.len()
            );
            (hits, has_more)
        } else {
            let hybrid_started = Instant::now();
            let mode = if semantic_only { "semantic" } else { "hybrid" };
            eprintln!(
                "[embedded-search] {mode} pipeline started candidate_limit={}",
                offset.saturating_add(limit).saturating_add(1).min(100)
            );
            let database = connect(&self.database_path)
                .execute()
                .await
                .map_err(internal_error)?;
            let pipeline = crate::hybrid_pipeline::HybridPipeline::new(
                &database,
                crate::hybrid_pipeline::TableNames {
                    chunks: crate::hybrid_pipeline::GLOBAL_CHUNK_TABLE.to_owned(),
                },
            );
            let embedder = self.embedder(model)?;
            let candidate_limit = offset.saturating_add(limit).saturating_add(1).min(100);
            let hybrid = if semantic_only {
                pipeline
                    .search_abstract_semantic(
                        &self.metadata_table.name(),
                        query,
                        model,
                        candidate_limit,
                        self.embedding_ctx,
                        &embedder,
                    )
                    .await
            } else {
                pipeline
                    .search_abstract(
                        &self.table,
                        &self.metadata_table.name(),
                        query,
                        model,
                        None,
                        candidate_limit,
                        self.embedding_ctx,
                        &embedder,
                    )
                    .await
            }
            .map_err(internal_error)?;
            eprintln!(
                "[embedded-search] {mode} pipeline complete elapsed_ms={} candidates={}",
                hybrid_started.elapsed().as_millis(),
                hybrid.hits.len()
            );
            let mut hits = Vec::new();
            for hit in hybrid.hits.into_iter().skip(offset).take(limit + 1) {
                let metadata = self.metadata_for_id(&hit.paper_id).await?;
                let (snippet, snippet_truncated) =
                    search_excerpt(&hit.text, query, SEARCH_SNIPPET_CHARS);
                hits.push(SearchHit {
                    paper_id: hit.paper_id,
                    metadata,
                    snippet,
                    snippet_truncated,
                });
            }
            let has_more = hits.len() > limit;
            hits.truncate(limit);
            eprintln!(
                "[embedded-search] result metadata complete returned={} has_more={has_more}",
                hits.len()
            );
            (hits, has_more)
        };
        let result_count = results.len();
        let next_offset = offset.saturating_add(result_count);
        let response = SearchResponse {
            search_url: format!(
                "/?tool=search_embedded_papers&embedded_only=1&query={}&embedding_model={}&limit={limit}&offset={offset}",
                Self::encode_query_parameter(query),
                Self::encode_query_parameter(model),
            ),
            metadata: SearchMetadata {
                lookup_time_ms: started_at.elapsed().as_millis() as u64,
                result_count,
                offset,
                next_offset,
                has_more,
            },
            results,
        };
        eprintln!(
            "[embedded-search] complete elapsed_ms={} result_count={} has_more={has_more}",
            started_at.elapsed().as_millis(),
            response.metadata.result_count
        );
        serde_json::to_string_pretty(&response).map_err(internal_error)
    }

    #[tool(
        description = "Report whether model-specific abstract embeddings exist and whether all of them are covered by the vector index."
    )]
    async fn abstract_embedding_index_status(
        &self,
        Parameters(args): Parameters<AbstractIndexArgs>,
    ) -> Result<String, McpError> {
        let model = args
            .embedding_model
            .as_deref()
            .unwrap_or(&self.embedding_model);
        if model != self.embedding_model {
            return Err(McpError::invalid_params(
                "embedding_model must match the configured local embedding model",
                None,
            ));
        }
        let metadata_table = self
            .open_fresh_metadata_table()
            .await
            .map_err(internal_error)?;
        let status = crate::hybrid_pipeline::abstract_vector_index_status(&metadata_table, model)
            .await
            .map_err(internal_error)?;
        serde_json::to_string(&status).map_err(internal_error)
    }

    #[tool(
        description = "Report abstract embedding and IVF-PQ index counts for every model with a vector column in the metadata table."
    )]
    async fn abstract_embedding_index_stats(&self) -> Result<String, McpError> {
        let metadata_table = self
            .open_fresh_metadata_table()
            .await
            .map_err(internal_error)?;
        let schema = metadata_table.schema().await.map_err(internal_error)?;
        let mut models = BTreeSet::from([self.embedding_model.clone()]);
        models.extend(
            schema
                .fields()
                .iter()
                .filter_map(|field| model_from_vector_column(field.name())),
        );
        let mut statuses = Vec::with_capacity(models.len());
        for model in models {
            statuses.push(
                crate::hybrid_pipeline::abstract_vector_index_status(&metadata_table, &model)
                    .await
                    .map_err(internal_error)?,
            );
        }
        serde_json::to_string_pretty(&statuses).map_err(internal_error)
    }

    #[tool(
        description = "Build or update the abstract vector index for the configured embedding model. This operation may take time and is separate from embedding."
    )]
    async fn build_abstract_vector_index(
        &self,
        Parameters(args): Parameters<AbstractIndexArgs>,
    ) -> Result<String, McpError> {
        let model = args
            .embedding_model
            .as_deref()
            .unwrap_or(&self.embedding_model);
        if model != self.embedding_model {
            return Err(McpError::invalid_params(
                "embedding_model must match the configured local embedding model",
                None,
            ));
        }
        let metadata_table = self
            .open_fresh_metadata_table()
            .await
            .map_err(internal_error)?;
        let status = crate::hybrid_pipeline::abstract_vector_index_status(&metadata_table, model)
            .await
            .map_err(internal_error)?;
        if !status.has_embeddings {
            return Err(McpError::invalid_params(
                "no indexed abstract embeddings are available yet",
                None,
            ));
        }
        if status.vector_index_ready {
            return serde_json::to_string(&status).map_err(internal_error);
        }

        let started_at = Instant::now();
        eprintln!(
            "[abstract-vector-index] GPU IVF-PQ build started model={model:?} embedded={} unindexed={}",
            status.embedded_vectors, status.unindexed_vectors
        );
        let table_name =
            env::var("ARXIV_METADATA_TABLE").unwrap_or_else(|_| "arxiv_metadata".to_owned());
        if let Err(error) = self.ensure_gpu_vector_index(&table_name, model).await {
            eprintln!(
                "[abstract-vector-index] GPU IVF-PQ build failed elapsed_ms={} error={error}",
                started_at.elapsed().as_millis()
            );
            return Err(internal_error(error));
        }
        let refreshed_metadata_table = self
            .open_fresh_metadata_table()
            .await
            .map_err(internal_error)?;
        let final_status =
            crate::hybrid_pipeline::abstract_vector_index_status(&refreshed_metadata_table, model)
                .await
                .map_err(internal_error)?;
        eprintln!(
            "[abstract-vector-index] complete elapsed_ms={} indexed={} unindexed={}",
            started_at.elapsed().as_millis(),
            final_status.indexed_vectors,
            final_status.unindexed_vectors
        );
        if !final_status.vector_index_ready {
            return Err(internal_error(format!(
                "vector index operation finished but {} vectors remain unindexed",
                final_status.unindexed_vectors
            )));
        }
        serde_json::to_string(&final_status).map_err(internal_error)
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
        let mode = args.mode.unwrap_or_else(|| "abstract".to_owned());
        if !matches!(mode.as_str(), "full_text" | "abstract") {
            return Err(McpError::invalid_params(
                "mode must be either full_text or abstract",
                None,
            ));
        }
        let job_id = self
            .index_jobs
            .start(paper_id.clone(), 0)
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
            match server.run_index_job(paper_id, token_budget, &mode).await {
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
        self.ensure_gpu_vector_index(table_name, &args.model)
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
        let mode = args.mode.as_deref().unwrap_or("abstract");
        if !matches!(mode, "full_text" | "abstract") {
            return Err(McpError::invalid_params(
                "mode must be either full_text or abstract",
                None,
            ));
        }
        let metadata_table_name =
            env::var("ARXIV_METADATA_TABLE").unwrap_or_else(|_| "arxiv_metadata".to_owned());
        if mode == "abstract" {
            let metadata_table = self
                .open_fresh_metadata_table()
                .await
                .map_err(internal_error)?;
            let status = crate::hybrid_pipeline::abstract_vector_index_status(
                &metadata_table,
                embedding_model,
            )
            .await
            .map_err(internal_error)?;
            if status.has_embeddings && !status.vector_index_ready {
                self.ensure_gpu_vector_index(&metadata_table_name, embedding_model)
                    .await
                    .map_err(internal_error)?;
            }
        } else {
            self.ensure_gpu_vector_index(table_name, embedding_model)
                .await
                .map_err(internal_error)?;
        }

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
        let reranker = reranker
            .as_ref()
            .map(|reranker| reranker as &dyn crate::test_pipeline::Reranker);
        let results = if mode == "abstract" {
            pipeline
                .search_abstract(
                    &self.table,
                    &metadata_table_name,
                    query,
                    embedding_model,
                    reranker,
                    candidate_limit,
                    ctx,
                    &embedder,
                )
                .await
        } else {
            pipeline
                .search(
                    query,
                    embedding_model,
                    reranker,
                    candidate_limit,
                    ctx,
                    &embedder,
                )
                .await
        }
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
    let index_build_progress = IndexBuildProgressHub::default();
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
        format!("http://127.0.0.1:{}", config.embedding.port),
        format!("http://127.0.0.1:{}", config.backend.port),
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
        index_build_progress.clone(),
    );

    let service: StreamableHttpService<ArxivMcpServer, LocalSessionManager> =
        StreamableHttpService::new(
            move || Ok(server.clone()),
            Arc::new(LocalSessionManager::default()),
            Default::default(),
        );
    let runtime_config = config.clone();
    let event_jobs = index_jobs.clone();
    let current_jobs = index_jobs.clone();
    let router = Router::new()
        .route("/health", axum::routing::get(|| async { "ok" }))
        .route(
            "/indexing-jobs/current",
            get(move || {
                let manager = current_jobs.clone();
                async move { Json(manager.current().await) }
            }),
        )
        .route(
            "/indexing-jobs/{job_id}/events",
            get(move |Path(job_id): Path<String>| {
                let manager = event_jobs.clone();
                async move { index_job_events(manager, job_id).await }
            }),
        )
        .route(
            "/index-builds/{job_id}/progress",
            axum::routing::post(
                move |Path(job_id): Path<String>, Json(update): Json<IndexBuildProgress>| {
                    let progress_hub = index_build_progress.clone();
                    async move {
                        if update.job_id != job_id {
                            return StatusCode::BAD_REQUEST;
                        }
                        match progress_hub.publish(update).await {
                            Ok(()) => StatusCode::ACCEPTED,
                            Err(error) => {
                                eprintln!(
                                    "[gpu-vector-index] rejected Python progress callback: {error}"
                                );
                                StatusCode::NOT_FOUND
                            }
                        }
                    }
                },
            ),
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
        "ArXiv search UI available at http://{}/",
        listener.local_addr()?
    );
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

fn model_from_vector_column(column: &str) -> Option<String> {
    let encoded = column.strip_prefix("embedding_")?;
    let bytes = encoded.as_bytes();
    if bytes.is_empty() || bytes.len() % 2 != 0 || !bytes.iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    let decoded = bytes
        .chunks_exact(2)
        .map(|pair| {
            let pair = std::str::from_utf8(pair).ok()?;
            u8::from_str_radix(pair, 16).ok()
        })
        .collect::<Option<Vec<_>>>()?;
    let model = String::from_utf8(decoded).ok()?;
    (!model.is_empty()).then_some(model)
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
            let finished = matches!(
                progress.state.as_str(),
                "complete" | "complete_with_errors" | "failed"
            );
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
        DEFAULT_EMBEDDING_CTX, IndexBuildProgress, IndexBuildProgressHub, IndexJobManager,
        MAX_LFM_EMBEDDING_TOKENS, SEARCH_SNIPPET_CHARS, ensure_indexing_state_fields,
        model_from_vector_column, pending_index_filter, search_excerpt, sql_string,
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

    #[tokio::test]
    async fn index_build_progress_is_pushed_to_registered_subscribers() {
        let hub = IndexBuildProgressHub::default();
        let mut updates = hub.subscribe("job-1").await.unwrap();
        hub.publish(IndexBuildProgress {
            job_id: "job-1".to_owned(),
            state: "completed".to_owned(),
            stage: "Index committed".to_owned(),
            completed: Some(1),
            total: Some(1),
            error: None,
        })
        .await
        .unwrap();

        let update = updates.recv().await.unwrap();
        assert_eq!(update.state, "completed");
        assert_eq!(update.stage, "Index committed");
    }

    #[test]
    fn discovers_model_names_from_embedding_vector_columns() {
        assert_eq!(
            model_from_vector_column("embedding_4c464d2d456d62656464696e672d51344b4d"),
            Some("LFM-Embedding-Q4KM".to_owned())
        );
        assert_eq!(model_from_vector_column("embedding_not_hex"), None);
        assert_eq!(model_from_vector_column("embedding_"), None);
    }

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
    async fn indexing_progress_covers_preparation_embedding_and_finalization() {
        let jobs = IndexJobManager::new();
        let job_id = jobs.start(None, 0).await.unwrap();
        let (queued, mut updates) = jobs.subscribe(&job_id).await.unwrap();
        assert_eq!(queued.state, "queued");
        assert!(!queued.total_papers_known);
        assert_eq!(queued.phase_total, None);
        assert!(jobs.start(None, 0).await.is_err());

        jobs.update(|progress| {
            progress.state = "running".to_owned();
            progress.phase = Some("Preparing abstract table".to_owned());
            progress.phase_completed = 2;
            progress.phase_total = Some(10);
        })
        .await;
        let preparing = updates.recv().await.unwrap();
        assert_eq!(preparing.phase_completed, 2);
        assert_eq!(preparing.phase_total, Some(10));
        assert_eq!(preparing.scanned_papers, 0);

        jobs.update(|progress| {
            progress.total_papers = 3;
            progress.total_papers_known = true;
            progress.phase_total = None;
            progress.phase_completed = 0;
            progress.scanned_papers = 3;
            progress.embedded_papers = 3;
            progress.phase = Some("Building abstract vector index".to_owned());
        })
        .await;
        let finalizing = updates.recv().await.unwrap();
        assert_eq!(finalizing.state, "running");
        assert_eq!(finalizing.scanned_papers, finalizing.total_papers);
        assert_eq!(finalizing.phase_total, None);
        assert_eq!(finalizing.eta_seconds, None);

        jobs.finish("complete", None).await;
        let complete = updates.recv().await.unwrap();
        assert_eq!(complete.state, "complete");
        let (reconnected, _) = jobs.subscribe(&job_id).await.unwrap();
        assert_eq!(reconnected.state, "complete");
        assert!(jobs.start(None, 0).await.is_ok());
    }

    #[tokio::test]
    async fn eta_is_available_after_one_embedding_sample() {
        let jobs = IndexJobManager::new();
        jobs.start(None, 10).await.unwrap();
        jobs.update(|progress| {
            progress.state = "running".to_owned();
            progress.total_papers_known = true;
            progress.scanned_papers = 1;
        })
        .await;
        jobs.record_embedding_completion(1).await;
        jobs.update(|_| {}).await;
        let progress = jobs.progress.read().await.clone().unwrap();
        assert!(progress.papers_per_second > 0.0);
        assert!(progress.eta_seconds.is_some());
    }

    #[tokio::test]
    async fn indexing_progress_preserves_errors_and_finishes_empty_jobs() {
        let jobs = IndexJobManager::new();
        let job_id = jobs.start(None, 0).await.unwrap();
        jobs.update(|progress| {
            progress.total_papers_known = true;
        })
        .await;
        jobs.finish("complete", None).await;
        let (empty, _) = jobs.subscribe(&job_id).await.unwrap();
        assert_eq!(empty.state, "complete");
        assert_eq!(empty.phase.as_deref(), Some("No papers pending indexing"));

        let job_id = jobs.start(None, 2).await.unwrap();
        jobs.update(|progress| {
            progress.failed_papers = 1;
            progress.error = Some("paper-1: embedding failed".to_owned());
        })
        .await;
        jobs.finish("complete", None).await;
        let (partial, _) = jobs.subscribe(&job_id).await.unwrap();
        assert_eq!(partial.state, "complete_with_errors");
        assert_eq!(partial.error.as_deref(), Some("paper-1: embedding failed"));

        let job_id = jobs.start(None, 0).await.unwrap();
        jobs.finish("failed", Some("preparation failed".to_owned()))
            .await;
        let (failed, _) = jobs.subscribe(&job_id).await.unwrap();
        assert_eq!(failed.state, "failed");
        assert_eq!(failed.error.as_deref(), Some("preparation failed"));
        assert!(!jobs.active.load(std::sync::atomic::Ordering::Acquire));
    }

    #[tokio::test]
    async fn indexing_job_pause_waits_for_resume() {
        let jobs = IndexJobManager::new();
        let job_id = jobs.start(None, 1).await.unwrap();
        jobs.update(|progress| progress.state = "running".to_owned())
            .await;
        jobs.update(|progress| progress.phase = Some("Building full-text vector index".to_owned()))
            .await;
        assert!(
            jobs.set_paused(&job_id, true)
                .await
                .unwrap_err()
                .contains("cannot be paused")
        );
        jobs.update(|progress| progress.phase = Some("embedding chunks".to_owned()))
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
