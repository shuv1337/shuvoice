//! Final-commit text post-processing helpers.

use std::collections::BTreeMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::types::TypingTextCase;

/// Compiled whole-word/phrase replacements (longest source first).
///
/// Uses Unicode-aware word characters (`alphanumeric || '_'`) and whole-phrase
/// boundaries without regex lookaround.
#[derive(Debug, Clone)]
pub struct CompiledReplacement {
    pub source: String,
    pub replacement: String,
}

/// Compiled replacement list shared across hot paths.
pub type CompiledTextReplacements = Arc<Vec<CompiledReplacement>>;

static MULTI_SPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r" {2,}").expect("multi-space regex"));
static LINE_BREAKS: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[ \t\f\v]*(?:\r\n|\r|\n)+[ \t\f\v]*").expect("line-break regex"));

/// Convert text to lowercase.
pub fn lowercase_text(text: &str) -> String {
    text.to_lowercase()
}

/// Capitalize the first alphabetic character, preserving leading spacing.
pub fn capitalize_first(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let mut chars: Vec<char> = text.chars().collect();
    for ch in &mut chars {
        if ch.is_alphabetic() {
            let upper = ch.to_uppercase().collect::<String>();
            if let Some(first) = upper.chars().next() {
                *ch = first;
            }
            break;
        }
    }
    chars.into_iter().collect()
}

/// Python `re` `\w` approximation under Unicode: letters/digits/underscore.
#[inline]
pub fn is_word_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

fn chars_eq_ignore_case(a: char, b: char) -> bool {
    if a == b {
        return true;
    }
    // Full Unicode case-folding comparison (handles non-ASCII letters).
    a.to_lowercase().eq(b.to_lowercase())
}

/// Find byte ranges of case-insensitive `needle` matches with `(?<!\w)...(?!\w)` boundaries.
pub fn find_bounded_phrase_matches(haystack: &str, needle: &str) -> Vec<(usize, usize)> {
    if needle.is_empty() || haystack.is_empty() {
        return Vec::new();
    }

    let hay: Vec<(usize, char)> = haystack.char_indices().collect();
    let needle_chars: Vec<char> = needle.chars().collect();
    if needle_chars.is_empty() || hay.len() < needle_chars.len() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    let last_start = hay.len() - needle_chars.len();
    let mut i = 0usize;
    while i <= last_start {
        let mut ok = true;
        for (offset, nc) in needle_chars.iter().enumerate() {
            if !chars_eq_ignore_case(hay[i + offset].1, *nc) {
                ok = false;
                break;
            }
        }
        if !ok {
            i += 1;
            continue;
        }

        let prev_is_word = i > 0 && is_word_char(hay[i - 1].1);
        let end_idx = i + needle_chars.len();
        let next_is_word = end_idx < hay.len() && is_word_char(hay[end_idx].1);
        if prev_is_word || next_is_word {
            i += 1;
            continue;
        }

        let start_byte = hay[i].0;
        let end_byte = if end_idx < hay.len() {
            hay[end_idx].0
        } else {
            haystack.len()
        };
        matches.push((start_byte, end_byte));
        // Non-overlapping left-to-right, same as re.sub default.
        i = end_idx;
    }
    matches
}

fn apply_one_replacement(text: &str, source: &str, replacement: &str) -> String {
    let ranges = find_bounded_phrase_matches(text, source);
    if ranges.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for (start, end) in ranges {
        out.push_str(&text[cursor..start]);
        out.push_str(replacement);
        cursor = end;
    }
    out.push_str(&text[cursor..]);
    out
}

/// Compile whole-word replacement entries once for hot-path reuse.
pub fn compile_text_replacements(
    replacements: &BTreeMap<String, String>,
) -> CompiledTextReplacements {
    let mut items: Vec<CompiledReplacement> = replacements
        .iter()
        .filter(|(k, _)| !k.is_empty())
        .map(|(k, v)| CompiledReplacement {
            source: k.clone(),
            replacement: v.clone(),
        })
        .collect();
    // Deterministic longest-first order with a stable lexical tie-breaker.
    // for unequal lengths. Equal-length ties are broken by lexicographic source
    // order so hot-path application is stable across runs/platforms (Python 3
    // sort is stable but input dict order is insertion-dependent; we choose a
    // fully deterministic tie-break).
    items.sort_by(|a, b| {
        b.source
            .len()
            .cmp(&a.source.len())
            .then_with(|| a.source.cmp(&b.source))
    });
    Arc::new(items)
}

/// Apply custom whole-word/phrase replacements case-insensitively.
pub fn apply_text_replacements(
    text: &str,
    replacements: Option<&BTreeMap<String, String>>,
    compiled_replacements: Option<&CompiledTextReplacements>,
) -> String {
    if text.is_empty() {
        return text.to_string();
    }

    let owned: CompiledTextReplacements;
    let compiled = if let Some(c) = compiled_replacements {
        c
    } else if let Some(map) = replacements {
        owned = compile_text_replacements(map);
        &owned
    } else {
        return text.to_string();
    };

    if compiled.is_empty() {
        return text.to_string();
    }

    let mut result = text.to_string();
    for entry in compiled.iter() {
        result = apply_one_replacement(&result, &entry.source, &entry.replacement);
    }

    let collapsed = MULTI_SPACE.replace_all(&result, " ");
    collapsed.trim().to_string()
}

/// Replace line breaks with spaces for Enter-to-submit safety.
pub fn sanitize_final_injection_text(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    LINE_BREAKS.replace_all(text, " ").trim().to_string()
}

/// Options for rendering transcript text consistently for overlay + inject.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub text_case: TypingTextCase,
    pub auto_capitalize: bool,
    pub replacements: CompiledTextReplacements,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            text_case: TypingTextCase::Default,
            auto_capitalize: true,
            replacements: Arc::new(Vec::new()),
        }
    }
}

/// Render transcript text for preview/final output consistency.
pub fn render_transcript_text(text: &str, options: &RenderOptions) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let rendered = apply_text_replacements(text, None, Some(&options.replacements));
    if rendered.is_empty() {
        return rendered;
    }
    match options.text_case {
        TypingTextCase::Lowercase => lowercase_text(&rendered),
        TypingTextCase::Default if options.auto_capitalize => capitalize_first(&rendered),
        TypingTextCase::Default => rendered,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capitalize_first_basic_cases() {
        assert_eq!(capitalize_first("hello world"), "Hello world");
        assert_eq!(capitalize_first(" already spaced"), " Already spaced");
        assert_eq!(capitalize_first("123abc"), "123Abc");
    }

    #[test]
    fn capitalize_first_noop_cases() {
        assert_eq!(capitalize_first(""), "");
        assert_eq!(capitalize_first("12345"), "12345");
    }

    #[test]
    fn lowercase_text_basic_cases() {
        assert_eq!(lowercase_text("Hello, World!"), "hello, world!");
        assert_eq!(lowercase_text("ShuVoice"), "shuvoice");
        assert_eq!(lowercase_text(""), "");
    }

    #[test]
    fn apply_text_replacements_phrase_and_case_insensitive() {
        let mut replacements = BTreeMap::new();
        replacements.insert("shove voice".into(), "ShuVoice".into());
        replacements.insert("speech to text".into(), "speech-to-text".into());
        replacements.insert("hyper land".into(), "Hyprland".into());
        let text = "Shove Voice, the real-time speech to text overlay for Hyper Land";
        assert_eq!(
            apply_text_replacements(text, Some(&replacements), None),
            "ShuVoice, the real-time speech-to-text overlay for Hyprland"
        );
    }

    #[test]
    fn apply_text_replacements_requires_word_boundaries() {
        let mut replacements = BTreeMap::new();
        replacements.insert("land".into(), "terrain".into());
        assert_eq!(
            apply_text_replacements("hyperland land", Some(&replacements), None),
            "hyperland terrain"
        );
        assert_eq!(
            apply_text_replacements("wonderland is great", Some(&replacements), None),
            "wonderland is great"
        );
        assert_eq!(
            apply_text_replacements("the land is great", Some(&replacements), None),
            "the terrain is great"
        );
    }

    #[test]
    fn apply_text_replacements_deletion() {
        let mut replacements = BTreeMap::new();
        replacements.insert("um".into(), "".into());
        replacements.insert("uh".into(), "".into());
        assert_eq!(
            apply_text_replacements("this um thing", Some(&replacements), None),
            "this thing"
        );
        assert_eq!(
            apply_text_replacements("uh hello um world", Some(&replacements), None),
            "hello world"
        );
        assert_eq!(apply_text_replacements("um", Some(&replacements), None), "");
    }

    #[test]
    fn apply_text_replacements_treats_replacement_as_literal_text() {
        let mut replacements = BTreeMap::new();
        replacements.insert("token".into(), r"\1 literal".into());
        assert_eq!(
            apply_text_replacements("token", Some(&replacements), None),
            r"\1 literal"
        );
    }

    #[test]
    fn apply_text_replacements_noop_cases() {
        let empty = BTreeMap::new();
        assert_eq!(apply_text_replacements("", Some(&empty), None), "");
        assert_eq!(
            apply_text_replacements("hello", Some(&empty), None),
            "hello"
        );
        assert_eq!(apply_text_replacements("hello", None, None), "hello");
    }

    #[test]
    fn apply_text_replacements_longer_phrase_matched_first() {
        let mut replacements = BTreeMap::new();
        replacements.insert("new york".into(), "NYC".into());
        replacements.insert("new york city".into(), "NYC metro".into());
        assert_eq!(
            apply_text_replacements("visit new york city", Some(&replacements), None),
            "visit NYC metro"
        );
    }

    #[test]
    fn sanitize_final_injection_text_replaces_line_breaks() {
        assert_eq!(
            sanitize_final_injection_text("hello\nworld\r\nthere"),
            "hello world there"
        );
        assert_eq!(sanitize_final_injection_text("  a  \n  b  "), "a b");
    }

    #[test]
    fn unicode_phrase_boundaries_match_public_word_class() {
        let mut replacements = BTreeMap::new();
        // Cyrillic letters are word chars: should NOT match inside a longer word.
        replacements.insert("кот".into(), "cat".into());
        assert_eq!(
            apply_text_replacements("котёнок кот", Some(&replacements), None),
            "котёнок cat"
        );

        // Accented Latin: whole-word match is case-insensitive.
        replacements.clear();
        replacements.insert("café".into(), "coffee".into());
        assert_eq!(
            apply_text_replacements("Café time", Some(&replacements), None),
            "coffee time"
        );
        assert_eq!(
            apply_text_replacements("xCaféy", Some(&replacements), None),
            "xCaféy"
        );
    }

    #[test]
    fn equal_length_tie_is_lexicographic_and_deterministic() {
        // Two equal-length sources: "ab" and "cd". Lexicographic order applies
        // "ab" before "cd". Overlapping is impossible here; we just assert order
        // of application via a shared substring scenario with non-overlap.
        let mut replacements = BTreeMap::new();
        replacements.insert("bb".into(), "2".into());
        replacements.insert("aa".into(), "1".into());
        let compiled = compile_text_replacements(&replacements);
        assert_eq!(compiled[0].source, "aa");
        assert_eq!(compiled[1].source, "bb");
        // Longer still wins over equal-length group.
        replacements.insert("aaa".into(), "3".into());
        let compiled = compile_text_replacements(&replacements);
        assert_eq!(compiled[0].source, "aaa");
    }

    #[test]
    fn hyphen_is_not_a_word_char_like_python() {
        let mut replacements = BTreeMap::new();
        replacements.insert("land".into(), "terrain".into());
        // Python (?<!\w)land(?!\w) matches after hyphen.
        assert_eq!(
            apply_text_replacements("high-land", Some(&replacements), None),
            "high-terrain"
        );
    }
}
