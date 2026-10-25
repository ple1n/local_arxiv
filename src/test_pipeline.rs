//! Pluggable remote model adapters and the single-paper chunk/embed/rerank test pipeline.

use serde::Deserialize;
use std::{error::Error, future::Future, pin::Pin, sync::Arc};
use tokenizers::Tokenizer;

pub type PipelineResult<T> = Result<T, Box<dyn Error + Send + Sync>>;
pub type ModelFuture<'a, T> = Pin<Box<dyn Future<Output = PipelineResult<T>> + Send + 'a>>;

/// Adapter contract for any embedding provider.
pub trait Embedder: Send + Sync {
    fn model_name(&self) -> &str;
    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        task: EmbeddingTask,
        ctx: usize,
    ) -> ModelFuture<'a, Vec<Vec<f32>>>;
}

#[derive(Clone, Copy)]
pub enum EmbeddingTask {
    Query,
    Document,
}

impl EmbeddingTask {
    fn prefix(self) -> &'static str {
        match self {
            Self::Query => "query: ",
            Self::Document => "document: ",
        }
    }
}

/// Adapter contract for any cross-encoder reranker.
pub trait Reranker: Send + Sync {
    fn model_name(&self) -> &str;
    fn rerank<'a>(&'a self, query: &'a str, documents: &'a [String]) -> ModelFuture<'a, Vec<f32>>;
}

#[derive(Clone)]
pub struct HttpEmbedder {
    client: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    model: String,
    tokenizer: Arc<Tokenizer>,
    max_tokens: usize,
}

impl HttpEmbedder {
    pub fn new(
        model: impl Into<String>,
        endpoint: impl Into<String>,
        tokenizer: Arc<Tokenizer>,
        max_tokens: usize,
    ) -> PipelineResult<Self> {
        if max_tokens < 2 {
            return Err("embedding max_tokens must be at least 2".into());
        }
        Ok(Self {
            client: reqwest::Client::builder().build()?,
            endpoint: endpoint.into(),
            api_key: std::env::var("EMBEDDING_API_KEY").ok(),
            model: model.into(),
            tokenizer,
            max_tokens,
        })
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Option<Vec<EmbeddingData>>,
    embeddings: Option<Vec<Vec<f32>>>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl Embedder for HttpEmbedder {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        task: EmbeddingTask,
        ctx: usize,
    ) -> ModelFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            let max_tokens = ctx.min(self.max_tokens);
            if max_tokens < 2 {
                return Err("embedding ctx must be at least 2 tokens".into());
            }
            let inputs = texts
                .iter()
                .map(|text| prepare_embedding_input(&self.tokenizer, text, task, max_tokens))
                .collect::<PipelineResult<Vec<_>>>()?;
            let mut request = self.client.post(&self.endpoint).json(&serde_json::json!({
                "model": self.model,
                "input": inputs,
            }));
            if let Some(key) = &self.api_key {
                request = request.bearer_auth(key);
            }
            let response: EmbeddingResponse =
                request.send().await?.error_for_status()?.json().await?;
            let mut embeddings = response
                .data
                .map(|data| data.into_iter().map(|item| item.embedding).collect())
                .or(response.embeddings)
                .ok_or("embedding service response has neither data nor embeddings")?;
            if embeddings.len() != texts.len() {
                return Err(format!(
                    "embedding service returned {} vectors for {} texts",
                    embeddings.len(),
                    texts.len()
                )
                .into());
            }
            for embedding in &mut embeddings {
                let norm = embedding
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt();
                if !norm.is_finite() || norm == 0.0 {
                    return Err("embedding service returned a zero or non-finite vector".into());
                }
                for value in embedding {
                    *value /= norm;
                }
            }
            Ok(embeddings)
        })
    }
}

fn prepare_embedding_input(
    tokenizer: &Tokenizer,
    text: &str,
    task: EmbeddingTask,
    max_tokens: usize,
) -> PipelineResult<String> {
    let prefixed = format!("{}{text}", task.prefix());
    let encoding = tokenizer.encode(prefixed.as_str(), true)?;
    if encoding.get_ids().len() <= max_tokens {
        return Ok(prefixed);
    }

    let ids = encoding.get_ids();
    for keep_tokens in (2..max_tokens).rev() {
        let candidate = tokenizer.decode(&ids[..keep_tokens], true)?;
        let candidate_encoding = tokenizer.encode(candidate.as_str(), true)?;
        if candidate_encoding.get_ids().len() <= max_tokens {
            return Ok(candidate);
        }
    }
    Err("embedding ctx is too small to preserve the task prefix".into())
}

#[derive(Clone)]
pub struct HttpReranker {
    client: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    model: String,
}

impl HttpReranker {
    pub fn from_env() -> PipelineResult<Self> {
        Self::from_env_with_model(None)
    }

    pub fn from_env_with_model(model: Option<&str>) -> PipelineResult<Self> {
        let endpoint = std::env::var("RERANK_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:9931/rerank".to_owned());
        Self::from_endpoint_with_model(model, endpoint)
    }

    pub fn from_endpoint_with_model(
        model: Option<&str>,
        endpoint: impl Into<String>,
    ) -> PipelineResult<Self> {
        Ok(Self {
            client: reqwest::Client::builder().build()?,
            endpoint: endpoint.into(),
            api_key: std::env::var("RERANK_API_KEY").ok(),
            model: model.map(str::to_owned).unwrap_or_else(|| {
                std::env::var("RERANK_MODEL")
                    .unwrap_or_else(|_| "Qwen3-Reranker-8B-Q4_K_M-32K".to_owned())
            }),
        })
    }
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankItem>,
}

#[derive(Deserialize)]
struct RerankItem {
    index: usize,
    relevance_score: f32,
}

impl Reranker for HttpReranker {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn rerank<'a>(&'a self, query: &'a str, documents: &'a [String]) -> ModelFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let mut request = self.client.post(&self.endpoint).json(&serde_json::json!({
                "model": self.model,
                "query": query,
                "documents": documents,
            }));
            if let Some(key) = &self.api_key {
                request = request.bearer_auth(key);
            }
            let response: RerankResponse = request.send().await?.error_for_status()?.json().await?;
            if response.results.len() != documents.len() {
                return Err(format!(
                    "rerank service returned {} scores for {} documents",
                    response.results.len(),
                    documents.len()
                )
                .into());
            }
            let mut scores = vec![None; documents.len()];
            for item in response.results {
                if item.index >= documents.len() || scores[item.index].is_some() {
                    return Err("reranker returned invalid or duplicate result indices".into());
                }
                scores[item.index] = Some(item.relevance_score);
            }
            scores
                .into_iter()
                .map(|score| score.ok_or_else(|| "reranker omitted a document score".into()))
                .collect()
        })
    }
}
