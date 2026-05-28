"""Unit tests for src/dataset_processing/text_cleaning.py.

Covers each private helper in isolation plus the two public entry points
(clean_fields, clean_text) end-to-end on representative dirty strings. The
edge cases here — empty input, decomposed unicode, CSV-escaped quotes, long
runs of the same character — are exactly the kind of thing that silently
regresses when the cleaning pipeline is touched.
"""

from src.dataset_processing.text_cleaning import (
    clean_fields,
    clean_text,
    _collapse_repeated_chars,
    _collapse_whitespace,
    _fix_double_quotes,
    _normalize_unicode,
)


# ---------------------------------------------------------------------------
# _normalize_unicode
# ---------------------------------------------------------------------------


def test_normalize_unicode_composes_decomposed_accents():
    # "e" + combining acute accent → single composed "é"
    decomposed = "é"
    result = _normalize_unicode(decomposed)
    assert result == "é"
    assert len(result) == 1


def test_normalize_unicode_leaves_plain_ascii_untouched():
    assert _normalize_unicode("plain text") == "plain text"


def test_normalize_unicode_empty_string():
    assert _normalize_unicode("") == ""


# ---------------------------------------------------------------------------
# _fix_double_quotes
# ---------------------------------------------------------------------------


def test_fix_double_quotes_collapses_csv_escaped_pair():
    assert _fix_double_quotes('He said ""hello""') == 'He said "hello"'


def test_fix_double_quotes_no_pairs_unchanged():
    assert _fix_double_quotes('single " quote') == 'single " quote'


def test_fix_double_quotes_empty_string():
    assert _fix_double_quotes("") == ""


# ---------------------------------------------------------------------------
# _collapse_repeated_chars
# ---------------------------------------------------------------------------


def test_collapse_repeated_chars_long_run_collapses_to_three():
    assert _collapse_repeated_chars("!!!!!!") == "!!!"


def test_collapse_repeated_chars_exactly_three_unchanged():
    # The threshold is "more than 3 in a row" → 3 is left alone.
    assert _collapse_repeated_chars("!!!") == "!!!"


def test_collapse_repeated_chars_four_collapses_to_three():
    assert _collapse_repeated_chars("aaaa") == "aaa"


def test_collapse_repeated_chars_handles_letters_and_punctuation():
    # "soooo" (4 o's) → "sooo"; "gooood" (4 o's) → "goood"; "!!!!!" (5) → "!!!"
    assert _collapse_repeated_chars("soooo gooood!!!!!") == "sooo goood!!!"


def test_collapse_repeated_chars_empty_string():
    assert _collapse_repeated_chars("") == ""


# ---------------------------------------------------------------------------
# _collapse_whitespace
# ---------------------------------------------------------------------------


def test_collapse_whitespace_multiple_spaces():
    assert _collapse_whitespace("a    b") == "a b"


def test_collapse_whitespace_tabs_and_newlines():
    assert _collapse_whitespace("a\t\tb\n\nc") == "a b c"


def test_collapse_whitespace_strips_leading_and_trailing():
    assert _collapse_whitespace("   padded   ") == "padded"


def test_collapse_whitespace_empty_string():
    assert _collapse_whitespace("") == ""


def test_collapse_whitespace_only_whitespace_becomes_empty():
    assert _collapse_whitespace("   \t\n  ") == ""


# ---------------------------------------------------------------------------
# clean_fields — combines all helpers per field
# ---------------------------------------------------------------------------


def test_clean_fields_applies_full_pipeline_to_both():
    title, text = clean_fields('Greaaaaat   ""deal""', "Works\t\tperfectly!!!!!")
    assert title == 'Greaaat "deal"'
    assert text == "Works perfectly!!!"


def test_clean_fields_handles_none_inputs():
    # clean_fields coerces None to "" before processing.
    title, text = clean_fields(None, None)
    assert title == ""
    assert text == ""


def test_clean_fields_empty_strings():
    assert clean_fields("", "") == ("", "")


# ---------------------------------------------------------------------------
# clean_text — title + text merged into one embedding-ready string
# ---------------------------------------------------------------------------


def test_clean_text_combines_title_and_text():
    assert clean_text("Title", "Body") == "Title Body"


def test_clean_text_empty_title_drops_leading_space():
    assert clean_text("", "Body only") == "Body only"


def test_clean_text_empty_text_keeps_title():
    assert clean_text("Title only", "") == "Title only"


def test_clean_text_both_empty_returns_empty():
    assert clean_text("", "") == ""


def test_clean_text_collapses_whitespace_across_join():
    # Even if both fields are clean, the join + final collapse must not leave
    # a double space between them.
    assert clean_text("A ", " B") == "A B"


def test_clean_text_end_to_end_on_dirty_review():
    title = 'AMAZING!!!!!!'
    text = 'I loooove   it,  ""best"" purchase\n\never'
    result = clean_text(title, text)
    assert result == 'AMAZING!!! I looove it, "best" purchase ever'
