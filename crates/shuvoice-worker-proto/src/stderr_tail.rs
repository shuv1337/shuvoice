//! Bounded stderr capture with secret/transcript redaction.

use std::collections::VecDeque;

/// Default max bytes retained from worker stderr.
pub const DEFAULT_STDERR_TAIL_BYTES: usize = 8 * 1024;

/// Push bytes into a ring buffer capped at `max_bytes`.
pub fn push_tail(tail: &mut VecDeque<u8>, max_bytes: usize, chunk: &[u8]) {
    if max_bytes == 0 {
        tail.clear();
        return;
    }
    for &b in chunk {
        if tail.len() == max_bytes {
            tail.pop_front();
        }
        tail.push_back(b);
    }
}

/// Convert a byte tail to a lossy UTF-8 string and redact sensitive-looking spans.
#[must_use]
pub fn redact_stderr_tail(bytes: &[u8]) -> String {
    let lossy = String::from_utf8_lossy(bytes);
    redact_text(&lossy)
}

/// Redact common secret patterns, assignment values, JWTs, and transcript-looking lines.
///
/// Never intended to be a perfect DLP filter — keep routine logs from echoing API
/// keys, bearer tokens, long opaque payloads, or ASR/TTS text.
#[must_use]
pub fn redact_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for line in input.split_inclusive('\n') {
        out.push_str(&redact_line(line));
    }
    // Strip NULs / other non-text control chars except newline/tab.
    out.chars()
        .map(|c| match c {
            '\n' | '\t' | '\r' => c,
            c if c.is_control() => '�',
            c => c,
        })
        .collect()
}

fn redact_line(line: &str) -> String {
    let lower = line.to_ascii_lowercase();
    // Whole-line redaction for obvious secret / transcript assignments.
    let hot_keys = [
        "api_key",
        "apikey",
        "api-key",
        "authorization",
        "secret",
        "password",
        "passwd",
        "token",
        "bearer ",
        "sk-",
        "private_key",
        "private-key",
        "access_key",
        "secret_access",
        "credential",
        // Avoid echoing model transcripts from worker logs.
        "transcript",
        "partial_transcript",
        "final_transcript",
        "recognized_text",
        "utterance_text",
        "asr_text",
        "synth_text",
        "tts_text",
    ];
    if hot_keys.iter().any(|k| lower.contains(k)) {
        // Keep a short prefix for diagnostics (logger name / level) when present.
        if let Some(idx) = line.find(':') {
            let (head, _) = line.split_at(idx + 1);
            // Avoid retaining secret material that appeared before the first colon.
            if hot_keys
                .iter()
                .any(|k| head.to_ascii_lowercase().contains(k))
            {
                return "[REDACTED]\n".to_string();
            }
            return format!("{head} [REDACTED]\n");
        }
        return "[REDACTED]\n".to_string();
    }

    // Redact long base64/hex/JWT-looking runs (possible tokens or payloads).
    // Split on non-token chars but treat '.' as a segment boundary (JWT).
    let mut result = String::with_capacity(line.len());
    let mut current = String::new();
    for ch in line.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '=' | '-' | '_') {
            current.push(ch);
        } else {
            flush_token(&mut result, &current);
            current.clear();
            // Do not copy raw JWT dots between high-entropy segments as a join
            // that would reassemble secrets; still keep punctuation for structure.
            result.push(ch);
        }
    }
    flush_token(&mut result, &current);

    // Second pass: key=value assignments with medium-length values.
    redact_assignments(&result)
}

fn flush_token(out: &mut String, token: &str) {
    // JWT segments and API tokens are often 20+ chars of mixed alphabet.
    if token.len() >= 20 && looks_high_entropy(token) {
        out.push_str("[REDACTED_BLOB]");
    } else if token.len() >= 32 {
        // Long alpha-only or digit-only runs are still suspicious.
        out.push_str("[REDACTED_BLOB]");
    } else {
        out.push_str(token);
    }
}

fn looks_high_entropy(token: &str) -> bool {
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    let has_alpha = token.chars().any(|c| c.is_ascii_alphabetic());
    has_digit && has_alpha
}

fn redact_assignments(line: &str) -> String {
    // word=value or word: value where value looks secret-ish.
    let mut out = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Copy until '='.
        if bytes[i] == b'=' {
            // Look back for key start.
            let key_end = i;
            let mut key_start = i;
            while key_start > 0 {
                let c = bytes[key_start - 1];
                if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' {
                    key_start -= 1;
                } else {
                    break;
                }
            }
            let key = line.get(key_start..key_end).unwrap_or("");
            let key_l = key.to_ascii_lowercase();
            let sensitive = [
                "key",
                "token",
                "secret",
                "password",
                "passwd",
                "auth",
                "credential",
                "bearer",
            ]
            .iter()
            .any(|k| key_l.contains(k));
            out.push('=');
            i += 1;
            // Capture value run.
            let val_start = i;
            while i < bytes.len() {
                let c = bytes[i];
                if c.is_ascii_whitespace() || c == b';' || c == b',' || c == b'"' || c == b'\'' {
                    break;
                }
                i += 1;
            }
            let val = line.get(val_start..i).unwrap_or("");
            if sensitive && !val.is_empty() {
                out.push_str("[REDACTED]");
            } else if val.len() >= 20 && looks_high_entropy(val) {
                out.push_str("[REDACTED_BLOB]");
            } else {
                out.push_str(val);
            }
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_respects_cap() {
        let mut q = VecDeque::new();
        push_tail(&mut q, 4, b"abcdef");
        assert_eq!(q.iter().copied().collect::<Vec<_>>(), b"cdef");
    }

    #[test]
    fn redacts_api_key_lines() {
        let s = redact_text("info: api_key=sk-abc123 super secret\nnext line ok\n");
        assert!(s.contains("[REDACTED]"));
        assert!(!s.contains("sk-abc"));
        assert!(s.contains("next line ok"));
    }

    #[test]
    fn redacts_long_entropy_blobs() {
        let blob = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6";
        let s = redact_text(&format!("payload {blob} end\n"));
        assert!(s.contains("[REDACTED_BLOB]"), "got {s:?}");
        assert!(!s.contains(blob));
    }

    #[test]
    fn redacts_transcript_lines() {
        let s = redact_text("worker: partial_transcript text=hello world secret phrase\n");
        assert!(s.contains("[REDACTED]"));
        assert!(!s.contains("hello world"));
    }

    #[test]
    fn redacts_jwt_segments() {
        let seg = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9";
        let s = redact_text(&format!("auth {seg}.payloadseg001.signatureseg002\n"));
        assert!(!s.contains(seg), "got {s:?}");
    }

    #[test]
    fn redacts_assignment_values() {
        let s = redact_text("export OPENAI_KEY=sk-proj-ABCDEFGHIJKLMNOP\n");
        assert!(!s.contains("sk-proj"), "got {s:?}");
    }
}
