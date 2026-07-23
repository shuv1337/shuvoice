//! Sherpa-ONNX backends (feature = "sherpa").
//!
//! Uses the official `sherpa-onnx` 1.13.4 crate. FFI recognizer ownership is
//! confined to this module and accessed only through `&mut self` on the
//! backend (single-threaded call pattern).

mod backend;
mod download;
mod model;
mod parakeet;

pub use backend::{SHERPA_REQUIRED_SAMPLE_RATE_HZ, SherpaBackend};
pub use download::{
    ALLOWED_DOWNLOAD_HOSTS, DEFAULT_BODY_INACTIVITY_TIMEOUT, DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_EXTRACT_INSTALL_TIMEOUT, DEFAULT_MAX_DOWNLOAD_BYTES, DEFAULT_MAX_TAR_ENTRIES,
    DEFAULT_MAX_UNCOMPRESSED_BYTES, DEFAULT_OVERALL_TIMEOUT, DownloadHardening, DownloadOptions,
    IO_CHUNK_SIZE, MAX_REDIRECTS, download_model, download_model_with_hardening, host_is_allowed,
    path_is_unsafe, safe_extract_tar_bz2, safe_extract_tar_bz2_with_limits, validate_download_url,
};
pub use model::{
    DEFAULT_SHERPA_MODEL_NAME, ModelFiles, collect_model_files, is_model_dir_complete,
    pick_model_onnx, resolve_model_dir,
};
pub use parakeet::{
    encoder_has_window_size, looks_like_parakeet_config, looks_like_parakeet_name,
    parakeet_streaming_startup_error,
};
