//! Shared HTTP helpers for cloud/local OpenAI-compatible TTS providers.

use reqwest::StatusCode;

use crate::error::TtsError;

pub fn classify_status(provider: &str, status: StatusCode) -> TtsError {
    let code = status.as_u16();
    let message = match code {
        401 => format!("{provider} authentication failed (401)"),
        429 => format!("{provider} rate limit exceeded (429)"),
        500..=599 => format!("{provider} server error ({code})"),
        _ => format!("{provider} request failed ({code})"),
    };
    TtsError::http(message)
}

pub fn map_reqwest_error(provider: &str, err: reqwest::Error) -> TtsError {
    if err.is_timeout() {
        return TtsError::timed_out(format!("{provider} request timed out"));
    }
    if let Some(status) = err.status() {
        return classify_status(provider, status);
    }
    TtsError::http(format!("{provider} request failed: {err}"))
}

pub fn build_client(timeout: std::time::Duration) -> Result<reqwest::Client, TtsError> {
    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(timeout.min(std::time::Duration::from_secs(10)))
        .build()
        .map_err(|err| TtsError::http(format!("failed to build HTTP client: {err}")))
}
