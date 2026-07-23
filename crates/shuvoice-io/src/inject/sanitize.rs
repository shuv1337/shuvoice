//! Final injection text sanitizer.

use regex::Regex;
use std::sync::OnceLock;

fn line_break_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // CR/LF plus Unicode line/paragraph separators and NEL (U+0085).
    RE.get_or_init(|| {
        Regex::new(r"[ \t\f\v]*(?:\r\n|\r|\n|\u{0085}|\u{2028}|\u{2029})+[ \t\f\v]*")
            .expect("static regex")
    })
}

/// Return final STT text that is safe for Enter-to-submit prompt boxes.
///
/// Collapses newline / Unicode line-separator runs (and adjacent horizontal
/// whitespace) to a single space, then strips ends. Internal multi-space is
/// preserved.
#[must_use]
pub fn sanitize_final_injection_text(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let replaced = line_break_re().replace_all(text, " ");
    replaced.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_cr_lf_line_breaks() {
        assert_eq!(sanitize_final_injection_text("fix this\n"), "fix this");
        assert_eq!(
            sanitize_final_injection_text("hello\r\nworld"),
            "hello world"
        );
        assert_eq!(
            sanitize_final_injection_text("line one\rline two"),
            "line one line two"
        );
        assert_eq!(
            sanitize_final_injection_text("keep  internal  spacing"),
            "keep  internal  spacing"
        );
    }

    #[test]
    fn replaces_unicode_line_separators() {
        assert_eq!(
            sanitize_final_injection_text("a\u{2028}b\u{2029}c"),
            "a b c"
        );
        assert_eq!(sanitize_final_injection_text("a\u{0085}b"), "a b");
        assert_eq!(
            sanitize_final_injection_text("  hello\u{2028}\nworld  "),
            "hello world"
        );
    }

    #[test]
    fn empty_and_whitespace_only() {
        assert_eq!(sanitize_final_injection_text(""), "");
        assert_eq!(sanitize_final_injection_text("\n\n"), "");
        assert_eq!(sanitize_final_injection_text("\r\n  \t"), "");
    }
}
