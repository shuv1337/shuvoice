from __future__ import annotations

from shuvoice.postprocess import apply_text_replacements, capitalize_first


def test_capitalize_first_basic_cases():
    assert capitalize_first("hello world") == "Hello world"
    assert capitalize_first(" already spaced") == " Already spaced"
    assert capitalize_first("123abc") == "123Abc"


def test_capitalize_first_noop_cases():
    assert capitalize_first("") == ""
    assert capitalize_first("12345") == "12345"


# --- apply_text_replacements ---


def test_apply_text_replacements_phrase_and_case_insensitive():
    replacements = {
        "shove voice": "ShuVoice",
        "speech to text": "speech-to-text",
        "hyper land": "Hyprland",
    }
    text = "Shove Voice, the real-time speech to text overlay for Hyper Land"

    assert (
        apply_text_replacements(text, replacements)
        == "ShuVoice, the real-time speech-to-text overlay for Hyprland"
    )


def test_apply_text_replacements_requires_word_boundaries():
    """Must not match inside other words (PR 14 & 15 both test this)."""
    replacements = {"land": "terrain"}

    assert apply_text_replacements("hyperland land", replacements) == "hyperland terrain"
    assert apply_text_replacements("wonderland is great", replacements) == "wonderland is great"
    assert apply_text_replacements("the land is great", replacements) == "the terrain is great"


def test_apply_text_replacements_deletion():
    """Empty replacement value should delete the word and collapse spaces."""
    replacements = {"um": "", "uh": ""}

    assert apply_text_replacements("this um thing", replacements) == "this thing"
    assert apply_text_replacements("uh hello um world", replacements) == "hello world"
    assert apply_text_replacements("um", replacements) == ""


def test_apply_text_replacements_treats_replacement_as_literal_text():
    replacements = {"token": r"\1 literal"}

    assert apply_text_replacements("token", replacements) == r"\1 literal"


def test_apply_text_replacements_noop_cases():
    assert apply_text_replacements("", {"a": "b"}) == ""
    assert apply_text_replacements("hello", {}) == "hello"
    assert apply_text_replacements("hello", None) == "hello"


def test_apply_text_replacements_longer_phrase_matched_first():
    """Longer sources are matched before shorter overlapping ones."""
    replacements = {
        "new york": "NYC",
        "new york city": "NYC metro",
    }

    assert apply_text_replacements("visit new york city", replacements) == "visit NYC metro"
