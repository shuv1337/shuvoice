from __future__ import annotations

from shuvoice.postprocess import apply_text_replacements, capitalize_first


def test_capitalize_first_basic_cases():
    assert capitalize_first("hello world") == "Hello world"
    assert capitalize_first(" already spaced") == " Already spaced"
    assert capitalize_first("123abc") == "123Abc"


def test_capitalize_first_noop_cases():
    assert capitalize_first("") == ""
    assert capitalize_first("12345") == "12345"


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
    replacements = {"land": "LAND"}

    assert apply_text_replacements("wonderland is great", replacements) == "wonderland is great"
    assert apply_text_replacements("the land is great", replacements) == "the LAND is great"
