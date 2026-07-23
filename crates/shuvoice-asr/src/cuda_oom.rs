//! CUDA/ORT OOM classification — owned by `shuvoice-core`.

pub use shuvoice_core::looks_like_cuda_oom_error;
pub use shuvoice_core::types::CUDA_OOM_ERROR_MARKERS;

/// Alias kept for call sites that pass `dyn Error`.
#[must_use]
pub fn looks_like_cuda_oom_error_dyn(err: &dyn std::error::Error) -> bool {
    looks_like_cuda_oom_error(&err.to_string())
}

/// String form used throughout this crate historically.
#[must_use]
pub fn looks_like_cuda_oom_str(text: &str) -> bool {
    looks_like_cuda_oom_error(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_match_core() {
        assert!(looks_like_cuda_oom_str("CUBLAS_STATUS_ALLOC_FAILED"));
        assert!(looks_like_cuda_oom_str(
            "bfc_arena Failed to allocate memory"
        ));
        assert!(!looks_like_cuda_oom_str("MemcpyFromHost only"));
    }
}
