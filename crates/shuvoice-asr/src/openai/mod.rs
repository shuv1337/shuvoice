//! OpenAI Realtime transcription backend (feature = "openai").

mod backend;
mod protocol;

pub use backend::OpenAiRealtimeBackend;
pub use protocol::{
    OPENAI_REALTIME_SAMPLE_RATE, OPENAI_REALTIME_WS_URL_DEFAULT, OpenAiProtocolState,
    clear_input_buffer_payload, redact_openai_error,
};
