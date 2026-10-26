use crate::python_embedding::PythonEmbeddingClient;
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
    arrow::arrow_array::{
        Array, BooleanArray, FixedSizeListArray, Float16Array, LargeStringArray, StringArray,
    },
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
use tokio::sync::{Mutex, Notify, RwLock, Semaphore, broadcast};
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
};

const DEFAULT_DATABASE_PATH: &str = "./data/arxiv_lance";
const DEFAULT_TABLE_NAME: &str = "arxiv_full_text";
const MAX_SEARCH_RESULTS: usize = 1_000;
const SEARCH_SNIPPET_CHARS: usize = 600;
const MAX_PAPER_CHARS: usize = 1_000_000;
const DEFAULT_EMBEDDING_CTX: usize = 2048;
const MAX_LFM_EMBEDDING_TOKENS: usize = 512;
const DEFAULT_CHUNK_TOKEN_BUDGET: usize =
    MAX_LFM_EMBEDDING_TOKENS - crate::test_pipeline::EMBEDDING_TOKEN_HEADROOM;
const ABSTRACT_PERSIST_BATCH_SIZE: usize = 1024;
const SEARCH_REQUEST_CACHE_TTL: Duration = Duration::from_secs(30 * 60);

static NEXT_INDEX_JOB_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Serialize)]
struct IndexJobProgress {
    job_id: String,
    embedding_model: String,
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
        self.start_for_model(paper_id, String::new(), total_papers)
            .await
    }

    async fn start_for_model(
        &self,
        paper_id: Option<String>,
        embedding_model: String,
        total_papers: usize,
    ) -> Result<String, String> {
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
            embedding_model,
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

#[derive(Clone)]
struct JobPausableEmbedder {
    inner: Arc<dyn Embedder>,
    jobs: Option<IndexJobManager>,
}

impl Embedder for JobPausableEmbedder {
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
            if let Some(jobs) = &self.jobs {
                jobs.wait_until_resumed().await;
            }
            let vectors = self.inner.embed(texts, task, ctx).await?;
            if let Some(jobs) = &self.jobs {
                jobs.record_embedding_completion(texts.len()).await;
            }
            Ok(vectors)
        })
    }
}

fn take_abstract_persist_batch(
    pending: &mut Vec<(String, String)>,
    flush_tail: bool,
) -> Option<Vec<(String, String)>> {
    if pending.len() < ABSTRACT_PERSIST_BATCH_SIZE && !flush_tail {
        return None;
    }
    let count = pending.len().min(ABSTRACT_PERSIST_BATCH_SIZE);
    if count == 0 {
        return None;
    }
    Some(pending.drain(..count).collect())
}

async fn embed_and_persist_abstract_batch(
    table: &Table,
    papers: Vec<(String, String)>,
    embedding_model: &str,
    embedding_ctx: usize,
    job_embedder: &JobPausableEmbedder,
    index_jobs: &IndexJobManager,
    scanned: &mut usize,
    embedded_count: &mut usize,
    failed_count: &mut usize,
    last_progress_update: &mut Instant,
    single_paper: bool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    const EMBEDDING_BATCH_SIZE: usize = 32;
    let groups = papers
        .chunks(EMBEDDING_BATCH_SIZE)
        .map(|group| group.to_vec())
        .collect::<Vec<_>>();
    index_jobs
        .update(|progress| progress.phase = Some("embedding abstracts".to_owned()))
        .await;
    let mut embedding_tasks = tokio::task::JoinSet::new();
    for group in groups {
        let embedding_model = embedding_model.to_owned();
        let job_embedder = job_embedder.clone();
        embedding_tasks.spawn(async move {
            let vectors = crate::hybrid_pipeline::embed_abstract_vectors(
                &group,
                &embedding_model,
                embedding_ctx,
                &job_embedder,
            )
            .await;
            (group, vectors)
        });
    }
    let mut batch_papers = Vec::new();
    let mut batch_vectors = Vec::new();
    while let Some(joined) = embedding_tasks.join_next().await {
        let (group, vector_result) = joined?;
        let group_len = group.len();
        let current = group.last().map(|(id, _)| id.clone());
        match vector_result {
            Ok(vectors) => {
                batch_papers.extend(group);
                batch_vectors.extend(vectors);
            }
            Err(error) => {
                *failed_count += group.len();
                eprintln!("Abstract batch embedding failed: {error}");
                index_jobs
                    .update(|progress| progress.error = Some(error.to_string()))
                    .await;
            }
        }
        *scanned += group_len;
        if *scanned % 100 < EMBEDDING_BATCH_SIZE
            || single_paper
            || last_progress_update.elapsed().as_secs_f64() >= 1.0
        {
            index_jobs
                .update(|progress| {
                    progress.scanned_papers = *scanned;
                    progress.chunked_papers = 0;
                    progress.embedded_papers = *embedded_count;
                    progress.failed_papers = *failed_count;
                    progress.current_paper = current;
                })
                .await;
            *last_progress_update = Instant::now();
        }
    }
    if !batch_papers.is_empty() {
        let write_started = Instant::now();
        match crate::hybrid_pipeline::write_abstract_vectors(
            table,
            &batch_papers,
            batch_vectors,
            embedding_model,
        )
        .await
        {
            Ok(count) => {
                *embedded_count += count;
                eprintln!(
                    "[abstract-index] persisted vectors={} elapsed_ms={}",
                    count,
                    write_started.elapsed().as_millis()
                );
            }
            Err(error) => {
                *failed_count += batch_papers.len();
                eprintln!(
                    "[abstract-index] vector write failed vectors={} elapsed_ms={}: {error}",
                    batch_papers.len(),
                    write_started.elapsed().as_millis()
                );
                index_jobs
                    .update(|progress| progress.error = Some(error.to_string()))
                    .await;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AppConfig {
    backend: ListenerConfig,
    frontend: ListenerConfig,
    embedding: EmbeddingServiceConfig,
    llama: ListenerConfig,
    #[serde(rename = "pythonEmbeddingModel")]
    python_embedding_model: String,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EmbeddingServiceConfig {
    host: IpAddr,
    port: u16,
    max_concurrent_texts: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchArgs {
    #[schemars(description = "Words or phrase to find in indexed paper text")]
    query: String,
    #[schemars(description = "Database search scope, from 1 to 1000; defaults to 5")]
    limit: Option<usize>,
    #[schemars(description = "Zero-based result page; defaults to 0")]
    page: Option<usize>,
    #[schemars(description = "Number of results sent in one page; defaults to the search scope")]
    page_size: Option<usize>,
    #[schemars(description = "Legacy result offset, converted to a page index")]
    offset: Option<usize>,
    #[schemars(
        description = "Request ID returned by a previous page; continues that cached search"
    )]
    request_id: Option<String>,
    #[schemars(description = "Response profile: compact (default) or web")]
    profile: Option<String>,
    #[schemars(
        description = "Optional exact result fields; overrides profile. Fields: paper_id, snippet, snippet_truncated, metadata.title, metadata.authors, metadata.abstract, metadata.categories, metadata.primary_category, metadata.doi, metadata.journal_ref, metadata.license, embedding_vector_f16_bits, cosine_distance"
    )]
    fields: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct EmbeddedSearchArgs {
    #[schemars(
        description = "Optional search text; empty query browses papers with vectors for the selected embedding model"
    )]
    query: Option<String>,
    #[schemars(
        description = "Optional embedding model override. Omit to use google/embeddinggemma-300m; set only to search vectors from another configured model."
    )]
    embedding_model: Option<String>,
    #[schemars(description = "Database search scope, from 1 to 1000")]
    limit: Option<usize>,
    #[schemars(description = "Zero-based result page; defaults to 0")]
    page: Option<usize>,
    #[schemars(description = "Number of results sent in one page; defaults to the search scope")]
    page_size: Option<usize>,
    #[schemars(description = "Legacy result offset, converted to a page index")]
    offset: Option<usize>,
    #[schemars(
        description = "Use semantic-only abstract-vector search instead of hybrid lexical and semantic search"
    )]
    semantic_only: Option<bool>,
    #[schemars(
        description = "Request ID returned by a previous page; continues that cached search"
    )]
    request_id: Option<String>,
    #[schemars(description = "Response profile: compact (default) or web")]
    profile: Option<String>,
    #[schemars(
        description = "Optional exact result fields; overrides profile. Fields: paper_id, snippet, snippet_truncated, metadata.title, metadata.authors, metadata.abstract, metadata.categories, metadata.primary_category, metadata.doi, metadata.journal_ref, metadata.license, embedding_vector_f16_bits, cosine_distance"
    )]
    fields: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct AbstractIndexArgs {
    #[schemars(
        description = "Optional embedding model override. Omit to use google/embeddinggemma-300m; set only to check or build another configured model's index."
    )]
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
    #[schemars(description = "Optional model ID for the Python embedding server")]
    model: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct InspectAbstractVectorsArgs {
    #[schemars(
        description = "1 to 10 arXiv paper IDs whose stored abstract vectors should be returned"
    )]
    paper_ids: Vec<String>,
    #[schemars(description = "Optional query to embed once and compare with each stored vector")]
    query: Option<String>,
    #[schemars(description = "Embedding model; defaults to the configured Python model")]
    embedding_model: Option<String>,
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
    #[schemars(
        description = "Optional embedding model override; omit to use google/embeddinggemma-300m, and set only to select another configured model"
    )]
    embedding_model: Option<String>,
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

#[derive(Clone, Debug, Serialize)]
struct SearchHit {
    paper_id: String,
    metadata: Option<PaperMetadata>,
    snippet: String,
    snippet_truncated: bool,
    embedding_vector_f16_bits: Option<Vec<u16>>,
    cosine_distance: Option<f32>,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<SearchHit>,
    metadata: SearchMetadata,
    search_url: String,
    request_id: String,
}

#[derive(Debug, Serialize)]
struct SearchMetadata {
    lookup_time_ms: u64,
    result_count: usize,
    offset: usize,
    next_offset: usize,
    has_more: bool,
    total_results: usize,
}

#[derive(Clone, Debug, Serialize)]
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

struct SearchRequestCache {
    mode: String,
    query: String,
    model: String,
    scope: usize,
    result_set: Option<CachedSearchResultSet>,
    pages: HashMap<(usize, usize, String), String>,
}

struct CachedSearchResultSet {
    results: Vec<SearchHit>,
    search_url: String,
    lookup_time_ms: u64,
}

struct SearchRequestEntry {
    created_at: Instant,
    cache: Arc<Mutex<SearchRequestCache>>,
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
    python_embedding_model: String,
    python_embedding_client: PythonEmbeddingClient,
    python_indexer: PythonIndexClient,
    model_list_endpoint: String,
    reranker_endpoint: String,
    embedding_tokenizer: Arc<Tokenizer>,
    embedding_scheduler: Arc<Semaphore>,
    embedding_ctx: usize,
    embedding_max_tokens: usize,
    index_jobs: IndexJobManager,
    index_build_progress: IndexBuildProgressHub,
    search_requests: Arc<RwLock<HashMap<String, SearchRequestEntry>>>,
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
        python_embedding_model: String,
        python_embedding_endpoint: String,
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
            python_embedding_model: python_embedding_model.clone(),
            python_embedding_client: PythonEmbeddingClient::new(
                python_embedding_endpoint,
                python_embedding_model.clone(),
                embedding_scheduler.clone(),
            ),
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
            search_requests: Arc::new(RwLock::new(HashMap::new())),
            tool_router: Self::tool_router(),
        }
    }

    async fn search_request_cache(
        &self,
        mode: &str,
        query: &str,
        model: &str,
        scope: usize,
        requested_id: Option<&str>,
    ) -> Result<(String, Arc<Mutex<SearchRequestCache>>), McpError> {
        let request_id = search_request_id(mode, query, model, scope)?;
        if requested_id.is_some_and(|requested_id| requested_id != request_id) {
            return Err(McpError::invalid_params(
                "request_id does not match the search parameters",
                None,
            ));
        }

        let now = Instant::now();
        let mut requests = self.search_requests.write().await;
        requests.retain(|_, entry| now.duration_since(entry.created_at) < SEARCH_REQUEST_CACHE_TTL);
        let entry = requests.entry(request_id.clone()).or_insert_with(|| {
            let created_at = Instant::now();
            SearchRequestEntry {
                created_at,
                cache: Arc::new(Mutex::new(SearchRequestCache {
                    mode: mode.to_owned(),
                    query: query.to_owned(),
                    model: model.to_owned(),
                    scope,
                    result_set: None,
                    pages: HashMap::new(),
                })),
            }
        });
        if requested_id.is_some() {
            if entry.cache.lock().await.result_set.is_none() {
                return Err(McpError::invalid_params(
                    "search request context has expired; start a new search",
                    None,
                ));
            }
        }
        Ok((request_id, entry.cache.clone()))
    }

    fn embedder(&self, model: &str) -> Result<Box<dyn Embedder>, McpError> {
        if model == self.python_embedding_model {
            return Ok(Box::new(self.python_embedding_client.clone()));
        }
        if model != self.embedding_model {
            return Err(McpError::invalid_params(
                "embedding model is not configured",
                None,
            ));
        }
        let embedder = crate::test_pipeline::HttpEmbedder::new(
            model,
            &self.embedding_endpoint,
            self.embedding_tokenizer.clone(),
            self.embedding_max_tokens,
            self.embedding_scheduler.clone(),
        )
        .map_err(internal_error)?;
        Ok(Box::new(embedder))
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
        embedding_model: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.index_jobs
            .update(|progress| {
                progress.phase = Some("Connecting to database".to_owned());
            })
            .await;
        let database = connect(&self.database_path).execute().await?;
        if mode == "abstract" {
            return self
                .run_abstract_index_job(&database, paper_id, embedding_model)
                .await;
        }
        let filter = if let Some(paper_id) = &paper_id {
            format!("paper_id = {}", sql_string(paper_id))
        } else {
            pending_index_filter(embedding_model)
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
                        embedding_model,
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
                embedding_model,
            )
            .await?;
        }
        Ok(())
    }

    async fn run_abstract_index_job(
        &self,
        _database: &lancedb::Connection,
        paper_id: Option<String>,
        embedding_model: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let table = &self.metadata_table;
        self.index_jobs
            .update(|progress| {
                progress.phase = Some("Counting pending abstracts".to_owned());
                progress.phase_completed = 0;
                progress.phase_total = None;
            })
            .await;
        let ready_column = crate::hybrid_pipeline::embedding_ready_column_name(embedding_model);
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
        let embedder: Arc<dyn Embedder> = Arc::from(self.embedder(embedding_model)?);
        let job_embedder = JobPausableEmbedder {
            inner: embedder,
            jobs: Some(self.index_jobs.clone()),
        };
        let mut scanned = 0;
        let mut embedded_count = 0;
        let mut failed_count = 0;
        let mut last_progress_update = Instant::now();
        let mut pending_papers = Vec::with_capacity(ABSTRACT_PERSIST_BATCH_SIZE);
        while let Some(batch) = rows.next().await {
            let batch = batch?;
            for row in 0..batch.num_rows() {
                let id = string_at(&batch, "paper_id", row)?
                    .ok_or("abstract row has a null paper_id")?;
                let abstract_text = string_at(&batch, "abstract", row)?.unwrap_or_default();
                pending_papers.push((id, abstract_text));
                if let Some(papers) = take_abstract_persist_batch(&mut pending_papers, false) {
                    embed_and_persist_abstract_batch(
                        table,
                        papers,
                        embedding_model,
                        self.embedding_ctx,
                        &job_embedder,
                        &self.index_jobs,
                        &mut scanned,
                        &mut embedded_count,
                        &mut failed_count,
                        &mut last_progress_update,
                        paper_id.is_some(),
                    )
                    .await?;
                }
            }
        }
        if let Some(papers) = take_abstract_persist_batch(&mut pending_papers, true) {
            embed_and_persist_abstract_batch(
                table,
                papers,
                embedding_model,
                self.embedding_ctx,
                &job_embedder,
                &self.index_jobs,
                &mut scanned,
                &mut embedded_count,
                &mut failed_count,
                &mut last_progress_update,
                paper_id.is_some(),
            )
            .await?;
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
            self.ensure_gpu_vector_index(&table_name, embedding_model)
                .await?;
        }
        Ok(())
    }

    async fn index_one_paper(
        &self,
        paper_id: &str,
        ctx: usize,
        token_budget: usize,
        embedding_model: &str,
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
                    embedding_model,
                    ctx,
                    token_budget,
                    false,
                )
                .await?;
            if result.1 {
                self.ensure_gpu_vector_index(
                    crate::hybrid_pipeline::GLOBAL_CHUNK_TABLE,
                    embedding_model,
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
        embedding_model: &str,
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
        let same_model = indexed_model == Some(embedding_model);
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
                    ("embedding_model", sql_string(embedding_model)),
                    ("chunk_count", chunk_count.to_string()),
                    ("embedded_chunk_count", "0".to_owned()),
                    ("indexing_error", "NULL".to_owned()),
                ],
            )
            .await?;
        } else if embedded == Some(true) && same_model {
            let (total, _) = pipeline
                .paper_embedding_progress(paper_id, embedding_model)
                .await?;
            return Ok((total, true));
        } else if !same_model {
            update_paper_index_state(
                &self.table,
                paper_id,
                &[
                    ("embedded", "false".to_owned()),
                    ("embedding_model", sql_string(embedding_model)),
                    ("embedded_chunk_count", "0".to_owned()),
                    ("indexing_error", "NULL".to_owned()),
                ],
            )
            .await?;
        }

        let embedder: Arc<dyn Embedder> = Arc::from(self.embedder(embedding_model)?);
        let job_embedder = JobPausableEmbedder {
            inner: embedder,
            jobs: pauseable.then(|| self.index_jobs.clone()),
        };
        self.index_jobs
            .update(|progress| progress.phase = Some("embedding chunks".to_owned()))
            .await;
        let embedding = pipeline
            .embed_paper_chunks_without_index(paper_id, embedding_model, ctx, &job_embedder, 4)
            .await;
        match embedding {
            Ok(_) => {
                let (chunk_count, embedded_count) = pipeline
                    .paper_embedding_progress(paper_id, embedding_model)
                    .await?;
                update_paper_index_state(
                    &self.table,
                    paper_id,
                    &[
                        ("chunked", "true".to_owned()),
                        ("embedded", "true".to_owned()),
                        ("embedding_model", sql_string(embedding_model)),
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
                    .paper_embedding_progress(paper_id, embedding_model)
                    .await?;
                update_paper_index_state(
                    &self.table,
                    paper_id,
                    &[
                        ("chunked", "true".to_owned()),
                        ("embedded", "false".to_owned()),
                        ("embedding_model", sql_string(embedding_model)),
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
        let limit = args.limit.unwrap_or(5);
        if limit > MAX_SEARCH_RESULTS {
            return Err(McpError::invalid_params(
                format!("limit must not exceed {MAX_SEARCH_RESULTS}"),
                None,
            ));
        }
        let page_size = args.page_size.unwrap_or(limit);
        if page_size == 0 || page_size > MAX_SEARCH_RESULTS {
            return Err(McpError::invalid_params(
                format!("page_size must be between 1 and {MAX_SEARCH_RESULTS}"),
                None,
            ));
        }
        select_search_fields(args.profile.as_deref(), args.fields.as_deref())?;
        let page = args
            .page
            .or_else(|| args.offset.map(|offset| offset / page_size))
            .unwrap_or(0);
        let (request_id, request_cache) = self
            .search_request_cache("lexical", query, "", limit, args.request_id.as_deref())
            .await?;
        let mut request_cache = request_cache.lock().await;
        if request_cache.mode != "lexical"
            || request_cache.query != query
            || !request_cache.model.is_empty()
            || request_cache.scope != limit
        {
            return Err(internal_error(
                "search request context does not match its key",
            ));
        }
        if request_cache.result_set.is_none() {
            let mut stream = self
                .table
                .query()
                .full_text_search(FullTextSearchQuery::new(query.to_owned()))
                .select(Select::columns(&["paper_id", "text"]))
                .limit(limit)
                .offset(0)
                .execute()
                .await
                .map_err(internal_error)?;
            let mut raw_hits = Vec::new();
            while let Some(batch) = stream.next().await {
                let batch = batch.map_err(internal_error)?;
                for row in 0..batch.num_rows() {
                    let paper_id = string_value(batch.column(0).as_ref(), row)
                        .map_err(internal_error)?
                        .unwrap_or_default()
                        .to_owned();
                    let text = string_value(batch.column(1).as_ref(), row)
                        .map_err(internal_error)?
                        .unwrap_or_default()
                        .to_owned();
                    raw_hits.push((paper_id, text));
                }
            }
            let paper_ids = raw_hits
                .iter()
                .map(|(paper_id, _)| paper_id.clone())
                .collect::<Vec<_>>();
            let metadata = self.metadata_for_ids(&paper_ids).await?;
            let results = raw_hits
                .into_iter()
                .map(|(paper_id, text)| {
                    let (snippet, snippet_truncated) =
                        search_excerpt(&text, query, SEARCH_SNIPPET_CHARS);
                    SearchHit {
                        metadata: metadata.get(&paper_id).cloned(),
                        paper_id,
                        snippet,
                        snippet_truncated,
                        embedding_vector_f16_bits: None,
                        cosine_distance: None,
                    }
                })
                .collect();
            request_cache.result_set = Some(CachedSearchResultSet {
                results,
                search_url: format!(
                    "/?tool=search_full_text&query={}&limit={limit}",
                    Self::encode_query_parameter(query)
                ),
                lookup_time_ms: started_at.elapsed().as_millis() as u64,
            });
        }
        cached_search_page(
            &request_id,
            &mut request_cache,
            page,
            page_size,
            args.profile.as_deref(),
            args.fields.as_deref(),
        )
    }

    #[tool(
        description = "Search papers by abstract vectors using hybrid lexical+semantic search (default) or semantic-only search; an empty query browses papers embedded for the selected model. Omit embedding_model to use google/embeddinggemma-300m; set it only to select another configured model. limit is the total candidate scope (up to 1000); page_size controls how many results are returned per page."
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
        let limit = args.limit.unwrap_or(10);
        if limit == 0 {
            return Err(McpError::invalid_params("limit must be at least 1", None));
        }
        if limit > MAX_SEARCH_RESULTS {
            return Err(McpError::invalid_params(
                format!("limit must not exceed {MAX_SEARCH_RESULTS}"),
                None,
            ));
        }
        let page_size = args.page_size.unwrap_or(limit);
        if page_size == 0 || page_size > MAX_SEARCH_RESULTS {
            return Err(McpError::invalid_params(
                format!("page_size must be between 1 and {MAX_SEARCH_RESULTS}"),
                None,
            ));
        }
        select_search_fields(args.profile.as_deref(), args.fields.as_deref())?;
        let page = args
            .page
            .or_else(|| args.offset.map(|offset| offset / page_size))
            .unwrap_or(0);
        let model = args
            .embedding_model
            .as_deref()
            .unwrap_or(&self.python_embedding_model);
        let semantic_only = args.semantic_only.unwrap_or(false);
        if model.trim().is_empty() || model.len() > 512 {
            return Err(McpError::invalid_params("embedding_model is invalid", None));
        }
        let mode = if query.is_empty() {
            "browse"
        } else if semantic_only {
            "semantic"
        } else {
            "hybrid"
        };
        let (request_id, request_cache) = self
            .search_request_cache(mode, query, model, limit, args.request_id.as_deref())
            .await?;
        let mut request_cache = request_cache.lock().await;
        if request_cache.mode != mode
            || request_cache.query != query
            || request_cache.model != model
            || request_cache.scope != limit
        {
            return Err(internal_error(
                "search request context does not match its key",
            ));
        }
        if request_cache.result_set.is_none() {
            let ready_column = crate::hybrid_pipeline::embedding_ready_column_name(model);
            eprintln!(
                "[embedded-search] start request_id={request_id} mode={mode} scope={limit} model={model:?} query_chars={}",
                query.chars().count()
            );
            self.metadata_table
                .schema()
                .await
                .map_err(internal_error)?
                .field_with_name(&ready_column)
                .map_err(internal_error)?;

            let results = if query.is_empty() {
                let mut rows = self
                    .metadata_table
                    .query()
                    .only_if(format!("{ready_column} = true"))
                    .select(Select::columns(&["paper_id", "abstract"]))
                    .limit(limit)
                    .offset(0)
                    .execute()
                    .await
                    .map_err(internal_error)?;
                let mut raw_hits = Vec::new();
                while let Some(batch) = rows.next().await {
                    let batch = batch.map_err(internal_error)?;
                    for row in 0..batch.num_rows() {
                        let paper_id = string_at(&batch, "paper_id", row)
                            .map_err(internal_error)?
                            .ok_or_else(|| {
                                internal_error("embedded metadata row has null paper_id")
                            })?
                            .to_owned();
                        let abstract_text = string_at(&batch, "abstract", row)
                            .map_err(internal_error)?
                            .unwrap_or_default()
                            .to_owned();
                        raw_hits.push((paper_id, abstract_text));
                    }
                }
                let ids = raw_hits
                    .iter()
                    .map(|(paper_id, _)| paper_id.clone())
                    .collect::<Vec<_>>();
                let metadata = self.metadata_for_ids(&ids).await?;
                raw_hits
                    .into_iter()
                    .map(|(paper_id, abstract_text)| SearchHit {
                        metadata: metadata.get(&paper_id).cloned(),
                        paper_id,
                        snippet: abstract_text,
                        snippet_truncated: false,
                        embedding_vector_f16_bits: None,
                        cosine_distance: None,
                    })
                    .collect::<Vec<_>>()
            } else {
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
                let hybrid = if semantic_only {
                    pipeline
                        .search_abstract_semantic(
                            &self.metadata_table.name(),
                            query,
                            model,
                            0,
                            limit,
                            self.embedding_ctx,
                            embedder.as_ref(),
                        )
                        .await
                } else {
                    pipeline
                        .search_abstract(
                            &self.table,
                            &self.metadata_table.name(),
                            query,
                            model,
                            0,
                            None,
                            limit,
                            self.embedding_ctx,
                            embedder.as_ref(),
                        )
                        .await
                }
                .map_err(internal_error)?;
                let mut dense_hits = hybrid.hits;
                if semantic_only {
                    dense_hits.sort_by(|left, right| {
                        left.cosine_distance
                            .unwrap_or(f32::INFINITY)
                            .total_cmp(&right.cosine_distance.unwrap_or(f32::INFINITY))
                            .then(left.paper_id.cmp(&right.paper_id))
                    });
                }
                let ids = dense_hits
                    .iter()
                    .map(|hit| hit.paper_id.clone())
                    .collect::<Vec<_>>();
                let metadata = self.metadata_for_ids(&ids).await?;
                dense_hits
                    .into_iter()
                    .map(|hit| {
                        let (snippet, snippet_truncated) =
                            search_excerpt(&hit.text, query, SEARCH_SNIPPET_CHARS);
                        SearchHit {
                            metadata: metadata.get(&hit.paper_id).cloned(),
                            paper_id: hit.paper_id,
                            snippet,
                            snippet_truncated,
                            embedding_vector_f16_bits: hit.embedding_vector_f16_bits,
                            cosine_distance: hit.cosine_distance,
                        }
                    })
                    .collect::<Vec<_>>()
            };

            request_cache.result_set = Some(CachedSearchResultSet {
                results,
                search_url: format!(
                    "/?tool=search_embedded_papers&embedded_only=1&query={}&embedding_model={}&limit={limit}",
                    Self::encode_query_parameter(query),
                    Self::encode_query_parameter(model),
                ),
                lookup_time_ms: started_at.elapsed().as_millis() as u64,
            });
            eprintln!(
                "[embedded-search] pipeline complete elapsed_ms={} scope_results={}",
                started_at.elapsed().as_millis(),
                request_cache
                    .result_set
                    .as_ref()
                    .map_or(0, |result_set| result_set.results.len())
            );
        } else {
            eprintln!("[embedded-search] cache hit request_id={request_id}");
        }
        cached_search_page(
            &request_id,
            &mut request_cache,
            page,
            page_size,
            args.profile.as_deref(),
            args.fields.as_deref(),
        )
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
            .unwrap_or(&self.python_embedding_model);
        if model != self.embedding_model && model != self.python_embedding_model {
            return Err(McpError::invalid_params(
                "embedding_model must match a configured embedding provider",
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
        let mut models = BTreeSet::from([
            self.embedding_model.clone(),
            self.python_embedding_model.clone(),
        ]);
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
        description = "Build or update the abstract vector index for the configured embedding model. This modifies the database and may take time; ask the user and obtain explicit consent before building it."
    )]
    async fn build_abstract_vector_index(
        &self,
        Parameters(args): Parameters<AbstractIndexArgs>,
    ) -> Result<String, McpError> {
        let model = args
            .embedding_model
            .as_deref()
            .unwrap_or(&self.python_embedding_model);
        if model != self.embedding_model && model != self.python_embedding_model {
            return Err(McpError::invalid_params(
                "embedding_model must match a configured embedding provider",
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
        description = "Writes embeddings and index state to LanceDB for one paper or all pending papers; may also build vector indexes. Ask the user and obtain explicit consent before starting this job. Progress is streamed from the backend."
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
        let embedding_model = args
            .embedding_model
            .unwrap_or_else(|| self.python_embedding_model.clone());
        if embedding_model != self.python_embedding_model && embedding_model != self.embedding_model
        {
            return Err(McpError::invalid_params(
                "embedding_model must match a configured embedding provider",
                None,
            ));
        }
        let job_id = self
            .index_jobs
            .start_for_model(paper_id.clone(), embedding_model.clone(), 0)
            .await
            .map_err(|error| McpError::invalid_request(error, None))?;
        let server = self.clone();
        let jobs = self.index_jobs.clone();
        let running_id = job_id.clone();
        let selected_model = embedding_model.clone();
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
            match server
                .run_index_job(paper_id, token_budget, &mode, &embedding_model)
                .await
            {
                Ok(()) => jobs.finish("complete", None).await,
                Err(error) => jobs.finish("failed", Some(error.to_string())).await,
            };
            heartbeat.abort();
        });
        Ok(serde_json::json!({
            "job_id": running_id,
            "events_url": format!("/indexing-jobs/{running_id}/events"),
            "state": "queued",
            "embedding_model": selected_model,
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
        description = "List the configured Python and LFM embedding models and available llama-server reranking models."
    )]
    async fn list_models(&self) -> Result<String, McpError> {
        let endpoint =
            env::var("MODEL_LIST_URL").unwrap_or_else(|_| self.model_list_endpoint.clone());
        let mut request = reqwest::Client::new().get(endpoint);
        if let Ok(key) = env::var("EMBEDDING_API_KEY") {
            request = request.bearer_auth(key);
        }
        let mut available = AvailableModelsResponse {
            embedding: vec![
                AvailableModel {
                    id: self.embedding_model.clone(),
                    tags: vec!["embedding".to_owned(), "llama-server".to_owned()],
                },
                AvailableModel {
                    id: self.python_embedding_model.clone(),
                    tags: vec!["embedding".to_owned(), "python-server".to_owned()],
                },
            ],
            reranking: Vec::new(),
        };
        let response = async {
            request
                .send()
                .await?
                .error_for_status()?
                .json::<ModelListResponse>()
                .await
        }
        .await;
        match response {
            Ok(response) => {
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
            }
            Err(error) => eprintln!("Could not load llama-server reranker models: {error}"),
        }
        serde_json::to_string(&available).map_err(internal_error)
    }

    #[tool(
        description = "Embed 1 to 8 document passages with the specifically configured Python embedding model."
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
        let model = args
            .model
            .as_deref()
            .unwrap_or(&self.python_embedding_model);
        if model != self.python_embedding_model {
            return Err(McpError::invalid_params(
                "model must match the configured Python embedding model",
                None,
            ));
        }
        let embeddings = self
            .python_embedding_client
            .embed_documents(model, &args.texts)
            .await
            .map_err(internal_error)?;
        let dimensions = embeddings.first().map(Vec::len).unwrap_or(0);
        if dimensions == 0
            || embeddings
                .iter()
                .any(|embedding| embedding.len() != dimensions)
        {
            return Err(internal_error(
                "embedding service returned vectors with invalid values or dimensions",
            ));
        }
        serde_json::to_string(&serde_json::json!({
            "model": model,
            "count": embeddings.len(),
            "dimensions": dimensions,
            "normalization": "L2; cosine similarity equals dot product up to f16 storage rounding",
            "embeddings": embeddings,
        }))
        .map_err(internal_error)
    }

    #[tool(
        description = "Return already-stored abstract f16 vectors for paper IDs and optionally compute direct dot products and cosine similarities against one query embedding. Does not re-embed papers."
    )]
    async fn inspect_abstract_vectors(
        &self,
        Parameters(args): Parameters<InspectAbstractVectorsArgs>,
    ) -> Result<String, McpError> {
        if args.paper_ids.is_empty()
            || args.paper_ids.len() > 10
            || args
                .paper_ids
                .iter()
                .any(|id| id.trim().is_empty() || id.len() > 256)
        {
            return Err(McpError::invalid_params(
                "paper_ids must contain 1 to 10 non-empty paper IDs",
                None,
            ));
        }
        if args
            .query
            .as_ref()
            .is_some_and(|query| query.trim().is_empty() || query.len() > 2_000)
        {
            return Err(McpError::invalid_params(
                "query must be non-empty and at most 2000 bytes when provided",
                None,
            ));
        }
        let model = args
            .embedding_model
            .as_deref()
            .unwrap_or(&self.python_embedding_model);
        if model != self.embedding_model && model != self.python_embedding_model {
            return Err(McpError::invalid_params(
                "embedding_model must match a configured embedding provider",
                None,
            ));
        }
        let vector_column = crate::hybrid_pipeline::vector_column_name(model);
        let escaped_ids = args
            .paper_ids
            .iter()
            .map(|id| sql_string(id))
            .collect::<Vec<_>>()
            .join(", ");
        let mut rows = self
            .metadata_table
            .query()
            .only_if(format!("paper_id IN ({escaped_ids})"))
            .select(Select::columns(&["paper_id", &vector_column]))
            .execute()
            .await
            .map_err(internal_error)?;
        let mut stored = HashMap::new();
        while let Some(batch) = rows.next().await {
            let batch = batch.map_err(internal_error)?;
            let ids = batch
                .column_by_name("paper_id")
                .ok_or_else(|| internal_error("metadata table is missing paper_id"))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| internal_error("paper_id must be a UTF-8 string"))?;
            let vectors = batch
                .column_by_name(&vector_column)
                .ok_or_else(|| internal_error("metadata table is missing the vector column"))?
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| internal_error("stored vectors must be fixed-size lists"))?;
            let values = vectors
                .values()
                .as_any()
                .downcast_ref::<Float16Array>()
                .ok_or_else(|| internal_error("stored vector values must be float16"))?;
            let dimensions = vectors.value_length() as usize;
            for row in 0..batch.num_rows() {
                if ids.is_null(row) || vectors.is_null(row) {
                    continue;
                }
                let start = row * dimensions;
                let vector_f16 = (0..dimensions)
                    .map(|index| values.value(start + index))
                    .collect::<Vec<_>>();
                let vector_f16_bits = vector_f16
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>();
                let vector_f32 = vector_f16
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>();
                stored.insert(ids.value(row).to_owned(), (vector_f32, vector_f16_bits));
            }
        }
        for id in &args.paper_ids {
            if !stored.contains_key(id) {
                return Err(internal_error(format!(
                    "no stored vector found for paper ID {id:?} in model {model:?}"
                )));
            }
        }
        let query_vector = if let Some(query) = args.query.as_deref() {
            let embeddings = self
                .embedder(model)?
                .embed(
                    &[query.to_owned()],
                    crate::test_pipeline::EmbeddingTask::Query,
                    self.embedding_ctx,
                )
                .await
                .map_err(internal_error)?;
            Some(
                embeddings
                    .into_iter()
                    .next()
                    .ok_or_else(|| internal_error("embedding service returned no query vector"))?,
            )
        } else {
            None
        };
        if let Some(query_vector) = &query_vector {
            if query_vector.iter().any(|value| !value.is_finite()) {
                return Err(internal_error("query vector contains non-finite values"));
            }
        }
        let results = args
            .paper_ids
            .iter()
            .map(|id| {
                let (vector, vector_f16_bits) =
                    stored.get(id).expect("requested IDs were validated");
                if let Some(query_vector) = &query_vector {
                    if query_vector.len() != vector.len() {
                        return Err(internal_error(format!(
                            "query dimension {} does not match stored vector dimension {}",
                            query_vector.len(),
                            vector.len()
                        )));
                    }
                }
                let (dot_product, cosine_similarity) = query_vector
                    .as_ref()
                    .map(|query| {
                        let dot = query.iter().zip(vector).map(|(a, b)| a * b).sum::<f32>();
                        let query_norm =
                            query.iter().map(|value| value * value).sum::<f32>().sqrt();
                        let vector_norm =
                            vector.iter().map(|value| value * value).sum::<f32>().sqrt();
                        let cosine = if query_norm == 0.0 || vector_norm == 0.0 {
                            None
                        } else {
                            Some(dot / (query_norm * vector_norm))
                        };
                        (Some(dot), cosine)
                    })
                    .unwrap_or((None, None));
                Ok(serde_json::json!({
                    "paper_id": id,
                    "dimensions": vector.len(),
                    "stored_vector_f16_bits": vector_f16_bits,
                    "dot_product": dot_product,
                    "cosine_similarity": cosine_similarity,
                }))
            })
            .collect::<Result<Vec<_>, McpError>>()?;
        serde_json::to_string_pretty(&serde_json::json!({
            "embedding_model": model,
            "stored_vector_dtype": "float16",
            "query_vector": query_vector,
            "query_vector_dtype": query_vector.as_ref().map(|_| "float32"),
            "results": results,
            "math_note": "Dot product and cosine are computed over returned f16-stored vectors promoted to f32; query embedding is computed once.",
        }))
        .map_err(internal_error)
    }

    #[tool(
        description = "Writes embeddings for up to 8 chunk IDs in a chunk table. Ask the user and obtain explicit consent before modifying the database. Prefer start_indexing_job for resumable paper or corpus indexing."
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
        if args.model != self.embedding_model && args.model != self.python_embedding_model {
            return Err(McpError::invalid_params(
                "model must match a configured embedding provider",
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
            .embed_chunk_batch(&args.chunk_ids, &args.model, ctx, embedder.as_ref())
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
        description = "Create the model-specific vector index after chunk embeddings have been persisted. This modifies the database; ask the user and obtain explicit consent before creating the index."
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
        description = "Search chunks in a named LanceDB table using lexical and dense retrieval, reciprocal-rank fusion, and optional reranking. If the required vector index is missing, this tool may create it; obtain explicit user consent before invoking it when index creation may be needed."
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
                    0,
                    reranker,
                    candidate_limit,
                    ctx,
                    embedder.as_ref(),
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
                    embedder.as_ref(),
                )
                .await
        }
        .map_err(internal_error)?;
        serde_json::to_string(&results).map_err(internal_error)
    }

    #[tool(
        description = "Index or resume a paper in LanceDB, writing embedding and index state, then test it with lexical, dense, reciprocal-rank-fusion, and optional reranking. Ask the user and obtain explicit consent before invoking this database-modifying operation."
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
        if embedding_model != self.embedding_model && embedding_model != self.python_embedding_model
        {
            return Err(McpError::invalid_params(
                "embedding_model must match a configured embedding provider",
                None,
            ));
        }
        self.index_one_paper(paper_id, ctx, DEFAULT_CHUNK_TOKEN_BUDGET, embedding_model)
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
                embedder.as_ref(),
            )
            .await
            .map_err(internal_error)?;
        serde_json::to_string_pretty(&serde_json::json!({
            "paper_id": paper_id,
            "result": result,
        }))
        .map_err(internal_error)
    }

    async fn metadata_for_ids(
        &self,
        paper_ids: &[String],
    ) -> Result<HashMap<String, PaperMetadata>, McpError> {
        if paper_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let ids = paper_ids
            .iter()
            .map(|paper_id| sql_string(paper_id))
            .collect::<Vec<_>>()
            .join(", ");
        let mut stream = self
            .metadata_table
            .query()
            .only_if(format!("paper_id IN ({ids})"))
            .select(Select::columns(&[
                "paper_id",
                "title",
                "authors",
                "abstract",
                "categories",
                "primary_category",
                "doi",
                "journal_ref",
                "license",
            ]))
            .execute()
            .await
            .map_err(internal_error)?;
        let mut metadata = HashMap::with_capacity(paper_ids.len());
        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(internal_error)?;
            for row in 0..batch.num_rows() {
                let column = |index| -> Result<Option<String>, McpError> {
                    string_value(batch.column(index).as_ref(), row)
                        .map(|value| value.map(str::to_owned))
                        .map_err(internal_error)
                };
                let paper_id =
                    column(0)?.ok_or_else(|| internal_error("metadata row has null paper_id"))?;
                metadata.insert(
                    paper_id,
                    PaperMetadata {
                        title: column(1)?,
                        authors: column(2)?,
                        abstract_text: column(3)?,
                        categories: column(4)?,
                        primary_category: column(5)?,
                        doi: column(6)?,
                        journal_ref: column(7)?,
                        license: column(8)?,
                    },
                );
            }
        }
        Ok(metadata)
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
            "Search the arXiv data currently loaded in the local LanceDB tables; this may be a sample, not the complete arXiv dataset. \
                 Use search_full_text for lexical full-text search. Use search_embedded_papers for abstract-vector search: hybrid lexical+semantic is the default, and semantic_only=true requests vector-only ranking. \
                 Omit embedding_model to use google/embeddinggemma-300m; set it only to search another configured model's vectors. \
                 For embedded search, limit sets the total candidate scope (up to 1000), while page_size sets the number returned per page. \
                 Use dataset_status to inspect the loaded table row counts. \
                 Before any database-modifying operation, including embedding/indexing jobs, vector-index construction, or tools that may create a missing index, ask the user and obtain explicit consent. Do not infer consent from a request to search or inspect data.",
        )
    }
}

pub async fn run() -> Result<(), Box<dyn Error>> {
    let config: AppConfig = serde_json::from_slice(&std::fs::read("config.json")?)?;
    if config.python_embedding_model.trim().is_empty()
        || !(32..=4096).contains(&config.embedding.max_concurrent_texts)
        || config.models.embedding.id.trim().is_empty()
        || config.models.embedding.repo.trim().is_empty()
        || config.models.embedding.path.trim().is_empty()
        || config.models.embedding.tokenizer_path.trim().is_empty()
        || !(crate::test_pipeline::EMBEDDING_TOKEN_HEADROOM + 2..=MAX_LFM_EMBEDDING_TOKENS)
            .contains(&config.models.embedding.max_tokens)
    {
        return Err(format!(
            "pythonEmbeddingModel must be non-empty; embedding.maxConcurrentTexts must be from 32 to 4096; models.embedding requires non-empty id, repo, path, and tokenizerPath; maxTokens must be from {} to {MAX_LFM_EMBEDDING_TOKENS}",
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
        config.python_embedding_model.clone(),
        format!(
            "http://{}",
            SocketAddr::new(config.embedding.host, config.embedding.port)
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
        Arc::new(Semaphore::new(config.embedding.max_concurrent_texts)),
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

fn search_request_id(
    mode: &str,
    query: &str,
    model: &str,
    scope: usize,
) -> Result<String, McpError> {
    let canonical = serde_json::to_vec(&(mode, query, model, scope)).map_err(internal_error)?;
    Ok(blake3::hash(&canonical).to_hex().to_string())
}

fn cached_search_page(
    request_id: &str,
    cache: &mut SearchRequestCache,
    page: usize,
    page_size: usize,
    profile: Option<&str>,
    fields: Option<&[String]>,
) -> Result<String, McpError> {
    let selected_fields = select_search_fields(profile, fields)?;
    let field_cache_key = selected_fields.join(",");
    let cache_key = (page_size, page, field_cache_key);
    if let Some(cached) = cache.pages.get(&cache_key) {
        return Ok(cached.clone());
    }
    let result_set = cache
        .result_set
        .as_ref()
        .ok_or_else(|| internal_error("search result cache is not populated"))?;
    let offset = page
        .checked_mul(page_size)
        .ok_or_else(|| McpError::invalid_params("page is too large", None))?;
    let total_results = result_set.results.len();
    let results = result_set
        .results
        .iter()
        .skip(offset)
        .take(page_size)
        .cloned()
        .collect::<Vec<_>>();
    let next_offset = offset.saturating_add(results.len());
    let response = SearchResponse {
        results,
        metadata: SearchMetadata {
            lookup_time_ms: result_set.lookup_time_ms,
            result_count: next_offset.saturating_sub(offset),
            offset,
            next_offset,
            has_more: next_offset < total_results,
            total_results,
        },
        search_url: format!(
            "{}&page_size={page_size}&page={page}",
            result_set.search_url
        ),
        request_id: request_id.to_owned(),
    };
    let mut serialized_response = serde_json::to_value(response).map_err(internal_error)?;
    if let Some(results) = serialized_response
        .get_mut("results")
        .and_then(serde_json::Value::as_array_mut)
    {
        let top_level_fields = selected_fields
            .iter()
            .filter(|field| !field.starts_with("metadata."))
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let include_metadata = selected_fields
            .iter()
            .any(|field| field.starts_with("metadata."));
        for result in results {
            if let Some(object) = result.as_object_mut() {
                object.retain(|key, _| {
                    top_level_fields.contains(key.as_str())
                        || (key == "metadata" && include_metadata)
                });
                if let Some(metadata) = object
                    .get_mut("metadata")
                    .and_then(serde_json::Value::as_object_mut)
                {
                    metadata.retain(|key, _| selected_fields.contains(&format!("metadata.{key}")));
                }
                if object
                    .get("metadata")
                    .and_then(serde_json::Value::as_object)
                    .is_some_and(serde_json::Map::is_empty)
                {
                    object.remove("metadata");
                }
            }
        }
    }
    let serialized = serde_json::to_string_pretty(&serialized_response).map_err(internal_error)?;
    cache.pages.insert(cache_key, serialized.clone());
    Ok(serialized)
}

fn select_search_fields(
    profile: Option<&str>,
    fields: Option<&[String]>,
) -> Result<Vec<String>, McpError> {
    const COMPACT_FIELDS: &[&str] = &[
        "paper_id",
        "snippet",
        "metadata.title",
        "metadata.authors",
        "metadata.categories",
        "metadata.primary_category",
        "cosine_distance",
    ];
    const WEB_FIELDS: &[&str] = &[
        "paper_id",
        "snippet",
        "snippet_truncated",
        "metadata.title",
        "metadata.authors",
        "metadata.abstract",
        "metadata.categories",
        "metadata.primary_category",
        "metadata.doi",
        "metadata.journal_ref",
        "metadata.license",
        "embedding_vector_f16_bits",
        "cosine_distance",
    ];
    const ALLOWED_FIELDS: &[&str] = WEB_FIELDS;

    if let Some(fields) = fields {
        let mut selected = Vec::with_capacity(fields.len());
        for field in fields {
            if !ALLOWED_FIELDS.contains(&field.as_str()) {
                return Err(McpError::invalid_params(
                    format!("unknown search result field: {field}"),
                    None,
                ));
            }
            if !selected.contains(field) {
                selected.push(field.clone());
            }
        }
        return Ok(selected);
    }

    let fields = match profile.unwrap_or("compact") {
        "compact" => COMPACT_FIELDS,
        "web" => WEB_FIELDS,
        other => {
            return Err(McpError::invalid_params(
                format!("profile must be 'compact' or 'web', got '{other}'"),
                None,
            ));
        }
    };
    Ok(fields.iter().map(|field| (*field).to_owned()).collect())
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
        ABSTRACT_PERSIST_BATCH_SIZE, DEFAULT_EMBEDDING_CTX, IndexBuildProgress,
        IndexBuildProgressHub, IndexJobManager, MAX_LFM_EMBEDDING_TOKENS, SEARCH_SNIPPET_CHARS,
        cached_search_page, ensure_indexing_state_fields, model_from_vector_column,
        pending_index_filter, search_excerpt, search_request_id, select_search_fields, sql_string,
        take_abstract_persist_batch, update_paper_index_state, validate_embedding_ctx,
    };
    use lancedb::{
        arrow::{
            arrow_array::{RecordBatch, StringArray},
            arrow_schema::{DataType, Field, Schema},
        },
        connect,
    };
    use std::{collections::HashMap, sync::Arc};

    #[test]
    fn search_request_id_is_stable_and_scope_specific() {
        let original = search_request_id("semantic", "vector search", "model-a", 20).unwrap();
        assert_eq!(
            original,
            search_request_id("semantic", "vector search", "model-a", 20).unwrap()
        );
        assert_ne!(
            original,
            search_request_id("semantic", "vector search", "model-a", 100).unwrap()
        );
        assert_ne!(
            original,
            search_request_id("hybrid", "vector search", "model-a", 20).unwrap()
        );
    }

    #[test]
    fn cached_search_pages_slice_results_without_rebuilding_the_search() {
        let mut cache = super::SearchRequestCache {
            mode: "semantic".to_owned(),
            query: "query".to_owned(),
            model: "model".to_owned(),
            scope: 3,
            result_set: Some(super::CachedSearchResultSet {
                results: ["paper-1", "paper-2", "paper-3"]
                    .into_iter()
                    .map(|paper_id| super::SearchHit {
                        paper_id: paper_id.to_owned(),
                        metadata: None,
                        snippet: String::new(),
                        snippet_truncated: false,
                        embedding_vector_f16_bits: None,
                        cosine_distance: None,
                    })
                    .collect(),
                search_url: "/?tool=search_embedded_papers".to_owned(),
                lookup_time_ms: 10,
            }),
            pages: HashMap::new(),
        };

        let page = cached_search_page("request-id", &mut cache, 1, 2, None, None).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&page).unwrap();
        assert_eq!(parsed["results"].as_array().unwrap().len(), 1);
        assert_eq!(parsed["results"][0]["paper_id"], "paper-3");
        assert_eq!(parsed["metadata"]["offset"], 2);
        assert_eq!(parsed["metadata"]["total_results"], 3);
        assert!(!parsed["metadata"]["has_more"].as_bool().unwrap());
        assert_eq!(
            cached_search_page("request-id", &mut cache, 1, 2, None, None).unwrap(),
            page
        );
    }

    #[test]
    fn compact_search_profile_omits_large_diagnostic_fields() {
        let compact = select_search_fields(None, None).unwrap();
        assert!(compact.contains(&"paper_id".to_owned()));
        assert!(compact.contains(&"metadata.title".to_owned()));
        assert!(!compact.contains(&"embedding_vector_f16_bits".to_owned()));
        assert!(!compact.contains(&"metadata.license".to_owned()));
        assert!(
            select_search_fields(Some("web"), None)
                .unwrap()
                .contains(&"embedding_vector_f16_bits".to_owned())
        );
        assert!(select_search_fields(None, Some(&["paper_id".to_owned()])).is_ok());
        assert!(select_search_fields(None, Some(&["unknown".to_owned()])).is_err());
    }

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
    fn abstract_persistence_batches_across_small_input_batches() {
        let mut pending = Vec::new();
        for index in 0..ABSTRACT_PERSIST_BATCH_SIZE {
            pending.push((index.to_string(), "abstract".to_owned()));
            let batch = take_abstract_persist_batch(&mut pending, false);
            if index + 1 < ABSTRACT_PERSIST_BATCH_SIZE {
                assert!(batch.is_none());
            } else {
                assert_eq!(batch.unwrap().len(), ABSTRACT_PERSIST_BATCH_SIZE);
            }
        }
        pending.push(("tail".to_owned(), "abstract".to_owned()));
        assert!(take_abstract_persist_batch(&mut pending, false).is_none());
        assert_eq!(
            take_abstract_persist_batch(&mut pending, true)
                .unwrap()
                .len(),
            1
        );
        assert!(pending.is_empty());
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
