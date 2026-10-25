use serde::{Deserialize, Serialize};
use std::{error::Error, time::Duration};
use tokio::sync::mpsc::UnboundedSender;

#[derive(Clone)]
pub struct PythonIndexClient {
    base_url: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct IndexJobRequest<'a> {
    job_id: &'a str,
    database_path: &'a str,
    table_name: &'a str,
    column_name: &'a str,
}

#[derive(Deserialize)]
struct IndexJobTicket {
    job_id: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct IndexBuildProgress {
    pub job_id: String,
    pub state: String,
    pub stage: String,
    pub completed: Option<usize>,
    pub total: Option<usize>,
    pub error: Option<String>,
}

impl PythonIndexClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn build_index(
        &self,
        job_id: &str,
        database_path: &str,
        table_name: &str,
        column_name: &str,
        progress: UnboundedSender<IndexBuildProgress>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let response = self
            .client
            .post(format!("{}/index-jobs", self.base_url))
            .json(&IndexJobRequest {
                job_id,
                database_path,
                table_name,
                column_name,
            })
            .timeout(Duration::from_secs(30))
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(format!("Python indexer returned HTTP {status}: {body}").into());
        }
        let ticket: IndexJobTicket = response.json().await?;
        if ticket.job_id != job_id {
            return Err(format!(
                "Python indexer returned unexpected job ID {:?}",
                ticket.job_id
            )
            .into());
        }

        loop {
            let response = self
                .client
                .get(format!("{}/index-jobs/{}", self.base_url, ticket.job_id))
                .timeout(Duration::from_secs(30))
                .send()
                .await?;
            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await?;
                return Err(format!("Python indexer status returned HTTP {status}: {body}").into());
            }
            let update: IndexBuildProgress = response.json().await?;
            if update.job_id != ticket.job_id {
                return Err(format!(
                    "Python indexer returned status for unexpected job ID {:?}",
                    update.job_id
                )
                .into());
            }
            let state = update.state.clone();
            progress
                .send(update.clone())
                .map_err(|_| "index progress receiver was dropped")?;
            match state.as_str() {
                "completed" => return Ok(()),
                "failed" => {
                    return Err(format!(
                        "Python IVF-PQ build failed: {}",
                        update.error.as_deref().unwrap_or("unknown error")
                    )
                    .into());
                }
                "queued" | "running" => tokio::time::sleep(Duration::from_secs(1)).await,
                unexpected => {
                    return Err(format!(
                        "Python indexer returned unexpected job state {unexpected:?}"
                    )
                    .into());
                }
            }
        }
    }
}
