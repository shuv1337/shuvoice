//! Streaming transcript merge helpers.

pub const MIN_OVERLAP_CHARS: usize = 8;
pub const MIN_OVERLAP_WORDS: usize = 2;

const REPETITION_MIN_WORDS: usize = 20;
const REPETITION_MAX_RUN: usize = 8;
const REPETITION_MAX_UNIQUE_RATIO: f64 = 0.2;

fn normalize_word(word: &str) -> String {
    word.trim_matches(|c: char| {
        matches!(
            c,
            '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}'
        )
    })
    .to_ascii_lowercase()
}

fn max_consecutive_run(words: &[String]) -> usize {
    if words.is_empty() {
        return 0;
    }
    let mut best = 1usize;
    let mut current = 1usize;
    let mut last = &words[0];
    for word in &words[1..] {
        if word == last {
            current += 1;
            best = best.max(current);
        } else {
            current = 1;
            last = word;
        }
    }
    best
}

fn is_pathological_repetition(text: &str) -> bool {
    let words: Vec<String> = text
        .split_whitespace()
        .map(normalize_word)
        .filter(|w| !w.is_empty())
        .collect();
    if words.len() < REPETITION_MIN_WORDS {
        return false;
    }
    let max_run = max_consecutive_run(&words);
    let unique = words
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let unique_ratio = unique as f64 / words.len().max(1) as f64;
    if max_run >= REPETITION_MAX_RUN {
        return true;
    }
    unique_ratio <= REPETITION_MAX_UNIQUE_RATIO
}

fn stitch_by_word_overlap(previous: &str, candidate: &str) -> Option<String> {
    let new_words: Vec<&str> = candidate.split_whitespace().collect();
    if new_words.is_empty() {
        return None;
    }

    let limit = new_words.len();
    // Equivalent to Python previous.rsplit(None, limit)
    let prev_parts = rsplit_whitespace_max(previous, limit);
    if prev_parts.is_empty() {
        return None;
    }

    let prev_tail_words: Vec<&str> = if prev_parts.len() > limit {
        prev_parts[1..].to_vec()
    } else {
        prev_parts
    };

    let min_words = MIN_OVERLAP_WORDS.max(1);
    let max_words = prev_tail_words.len().min(new_words.len());
    if max_words < min_words {
        return None;
    }

    let prev_tail_norm: Vec<String> = prev_tail_words[prev_tail_words.len() - max_words..]
        .iter()
        .map(|w| normalize_word(w))
        .collect();
    let new_head_norm: Vec<String> = new_words[..max_words]
        .iter()
        .map(|w| normalize_word(w))
        .collect();

    for overlap in (min_words..=max_words).rev() {
        if prev_tail_norm[prev_tail_norm.len() - overlap..] != new_head_norm[..overlap] {
            continue;
        }
        let suffix_words = &new_words[overlap..];
        if suffix_words.is_empty() {
            return None;
        }
        let glue = if previous.ends_with([' ', '\n', '\t']) {
            ""
        } else {
            " "
        };
        return Some(format!("{previous}{glue}{}", suffix_words.join(" ")));
    }
    None
}

/// Split from the right on whitespace, at most `max_splits` times.
fn rsplit_whitespace_max(text: &str, max_splits: usize) -> Vec<&str> {
    if max_splits == 0 {
        return vec![text];
    }
    let bytes = text.as_bytes();
    let mut parts_rev: Vec<&str> = Vec::new();
    let mut end = text.len();
    let mut splits = 0usize;
    let mut i = text.len();
    while i > 0 && splits < max_splits {
        i -= 1;
        if bytes[i].is_ascii_whitespace() {
            let start = i + 1;
            if start < end {
                parts_rev.push(&text[start..end]);
                splits += 1;
            }
            // skip whitespace run
            while i > 0 && bytes[i - 1].is_ascii_whitespace() {
                i -= 1;
            }
            end = i;
        }
    }
    if end > 0 {
        // trim trailing whitespace from remainder head
        let mut head_end = end;
        while head_end > 0 && bytes[head_end - 1].is_ascii_whitespace() {
            head_end -= 1;
        }
        if head_end > 0 {
            parts_rev.push(&text[..head_end]);
        }
    }
    parts_rev.reverse();
    parts_rev
}

/// Prefer stable cumulative transcript growth over regressions.
pub fn prefer_transcript(previous: &str, candidate: &str) -> String {
    let previous_raw = previous;
    let candidate_raw = candidate;

    let prev = previous_raw.trim();
    let new = candidate_raw.trim();

    if new.is_empty() {
        return previous_raw.to_string();
    }
    if is_pathological_repetition(new) {
        return previous_raw.to_string();
    }
    if prev.is_empty() {
        return candidate_raw.to_string();
    }
    if is_pathological_repetition(prev) {
        return candidate_raw.to_string();
    }
    if new.starts_with(prev) {
        return candidate_raw.to_string();
    }
    if prev.starts_with(new) {
        return previous_raw.to_string();
    }

    let max_overlap_chars = prev.len().min(new.len()).min(200);
    if max_overlap_chars >= MIN_OVERLAP_CHARS {
        for overlap in (MIN_OVERLAP_CHARS..=max_overlap_chars).rev() {
            if prev.ends_with(&new[..overlap]) {
                return format!("{prev}{}", &new[overlap..]);
            }
        }
    }

    if let Some(stitched) = stitch_by_word_overlap(previous_raw, candidate_raw) {
        return stitched;
    }

    if new.len() > prev.len() {
        return candidate_raw.to_string();
    }

    previous_raw.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_growth_prefers_candidate() {
        assert_eq!(prefer_transcript("hello", "hello world"), "hello world");
    }

    #[test]
    fn regression_rejection_prefers_previous() {
        assert_eq!(prefer_transcript("hello world", "hello"), "hello world");
    }

    #[test]
    fn empty_whitespace_handling() {
        assert_eq!(prefer_transcript("", "hello"), "hello");
        assert_eq!(prefer_transcript("hello", "   "), "hello");
    }

    #[test]
    fn overlap_stitching_behavior() {
        let previous = "the quick brown fox";
        let candidate = "brown fox jumps high";
        assert_eq!(
            prefer_transcript(previous, candidate),
            "the quick brown fox jumps high"
        );
    }

    #[test]
    fn false_positive_short_overlap_does_not_stitch() {
        let previous = "alpha123beta";
        let candidate = "123beta and gamma";
        assert_eq!(MIN_OVERLAP_CHARS, 8);
        assert_eq!(prefer_transcript(previous, candidate), candidate);
    }

    #[test]
    fn rewrite_acceptance_for_longer_contextual_candidate() {
        let previous = "quick brown";
        let candidate = "the quick brown dog jumped";
        assert_eq!(prefer_transcript(previous, candidate), candidate);
    }

    #[test]
    fn equal_length_divergent_determinism_keeps_previous() {
        let previous = "hello world";
        let candidate = "jello world";
        assert_eq!(previous.len(), candidate.len());
        assert_eq!(prefer_transcript(previous, candidate), previous);
    }

    #[test]
    fn overlap_threshold_at_eight_chars_stitches() {
        let previous = "alpha1234beta";
        let candidate = "1234beta and gamma";
        assert_eq!(
            prefer_transcript(previous, candidate),
            "alpha1234beta and gamma"
        );
    }

    #[test]
    fn word_overlap_stitches_shifted_context() {
        assert_eq!(MIN_OVERLAP_WORDS, 2);
        let previous = "But I would definitely like to see";
        let candidate = "to see how cross platform we can make it";
        assert_eq!(
            prefer_transcript(previous, candidate),
            "But I would definitely like to see how cross platform we can make it"
        );
    }

    #[test]
    fn word_overlap_requires_new_suffix() {
        let previous = "we are done now";
        let candidate = "are done now";
        assert_eq!(prefer_transcript(previous, candidate), previous);
    }

    #[test]
    fn pathological_repetition_candidate_is_rejected() {
        let candidate = "just ".repeat(120);
        let candidate = candidate.trim();
        assert_eq!(prefer_transcript("", candidate), "");
    }

    #[test]
    fn pathological_previous_can_be_replaced_by_sane_shorter_candidate() {
        let previous = "testing, ".repeat(120);
        let previous = previous.trim();
        let candidate = "testing moonshine provider";
        assert_eq!(prefer_transcript(previous, candidate), candidate);
    }

    #[test]
    fn short_repetition_is_kept() {
        assert_eq!(prefer_transcript("", "no no no no"), "no no no no");
    }
}
