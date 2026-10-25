use serde::{Deserialize, Serialize};
use std::{error::Error, time::Duration};

#[derive(Clone)]
pub struct PythonIndexClient {
    base_url: String,
    callback_base_url: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct IndexJobRequest<'a> {
    job_id: &'a str,
    database_path: &'a str,
    table_name: &'a str,
    column_name: &'a str,
    callback_url: &'a str,
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
    pub fn new(base_url: impl Into<String>, callback_base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            callback_base_url: callback_base_url.into().trim_end_matches('/').to_owned(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn start_index(
        &self,
        job_id: &str,
        database_path: &str,
        table_name: &str,
        column_name: &str,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let callback_url = format!("{}/index-builds/{job_id}/progress", self.callback_base_url);
        let response = self
            .client
            .post(format!("{}/index-jobs", self.base_url))
            .json(&IndexJobRequest {
                job_id,
                database_path,
                table_name,
                column_name,
                callback_url: &callback_url,
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
        Ok(())
    }
}
