use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::{error::Error, time::Duration};
use tokio::sync::Semaphore;

pub type EmbeddingResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Clone)]
pub struct PythonEmbeddingClient {
    endpoint: String,
    model: String,
    client: reqwest::Client,
    embedding_gate: Arc<Semaphore>,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    texts: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    model: String,
    embeddings: Vec<Vec<f32>>,
    dimensions: usize,
}

impl PythonEmbeddingClient {
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        embedding_gate: Arc<Semaphore>,
    ) -> Self {
        Self {
            endpoint: format!("{}/embed/document", endpoint.into().trim_end_matches('/')),
            model: model.into(),
            client: reqwest::Client::new(),
            embedding_gate,
        }
    }

    async fn embed_for_task(
        &self,
        model: &str,
        texts: &[String],
        task: crate::test_pipeline::EmbeddingTask,
    ) -> EmbeddingResult<Vec<Vec<f32>>> {
        if model != self.model {
            return Err(
                format!("Python embedding client is configured for {:?}", self.model).into(),
            );
        }
        let permit_count = u32::try_from(texts.len())
            .map_err(|_| "embedding batch is too large for the concurrency gate")?;
        let _permit = self
            .embedding_gate
            .clone()
            .acquire_many_owned(permit_count)
            .await
            .map_err(|error| format!("embedding concurrency gate closed: {error}"))?;
        let endpoint = self.endpoint.replace(
            "/embed/document",
            if matches!(task, crate::test_pipeline::EmbeddingTask::Query) {
                "/embed/query"
            } else {
                "/embed/document"
            },
        );
        let response = self
            .client
            .post(endpoint)
            .json(&EmbedRequest { model, texts })
            .timeout(Duration::from_secs(180))
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(format!("Python embedding server returned HTTP {status}: {body}").into());
        }
        let response: EmbedResponse = response.json().await?;
        if response.model != model {
            return Err(format!(
                "Python embedding server returned model {:?}, expected {:?}",
                response.model, model
            )
            .into());
        }
        if response.embeddings.len() != texts.len()
            || response.dimensions == 0
            || response.embeddings.iter().any(|vector| {
                vector.len() != response.dimensions || vector.iter().any(|value| !value.is_finite())
            })
        {
            return Err("Python embedding server returned invalid vectors".into());
        }
        Ok(response.embeddings)
    }

    pub async fn embed_documents(
        &self,
        model: &str,
        texts: &[String],
    ) -> EmbeddingResult<Vec<Vec<f32>>> {
        self.embed_for_task(model, texts, crate::test_pipeline::EmbeddingTask::Document)
            .await
    }
}

impl crate::test_pipeline::Embedder for PythonEmbeddingClient {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn embed<'a>(
        &'a self,
        texts: &'a [String],
        task: crate::test_pipeline::EmbeddingTask,
        _ctx: usize,
    ) -> crate::test_pipeline::ModelFuture<'a, Vec<Vec<f32>>> {
        Box::pin(self.embed_for_task(&self.model, texts, task))
    }
}
