"""Text cleaning utilities for Amazon Review data points."""

import re
import unicodedata


def clean_fields(title: str, text: str) -> tuple[str, str]:
    """Clean title and text fields independently."""
    title_clean = _normalize_unicode(title or "")
    title_clean = _fix_double_quotes(title_clean)
    title_clean = _collapse_repeated_chars(title_clean)
    title_clean = _collapse_whitespace(title_clean)

    text_clean = _normalize_unicode(text or "")
    text_clean = _fix_double_quotes(text_clean)
    text_clean = _collapse_repeated_chars(text_clean)
    text_clean = _collapse_whitespace(text_clean)
    return title_clean, text_clean


def clean_text(title: str, text: str) -> str:
    """Combine title and text into a single clean string for embedding."""
    title_clean, text_clean = clean_fields(title, text)
    combined = f"{title_clean} {text_clean}".strip()
    return _collapse_whitespace(combined)


def _normalize_unicode(s: str) -> str:
    """NFC normalization — keeps accented chars (è, ü...) in canonical form."""
    return unicodedata.normalize("NFC", s)


def _fix_double_quotes(s: str) -> str:
    """Convert CSV-escaped double quotes \"\" → \"."""
    return s.replace('""', '"')


def _collapse_repeated_chars(s: str) -> str:
    """Collapse any char repeated more than 3 times in a row (e.g. !!!!!! → !!!)."""
    return re.sub(r'(.)\1{3,}', lambda m: m.group(1) * 3, s)


def _collapse_whitespace(s: str) -> str:
    """Normalize multiple spaces/tabs/newlines to a single space."""
    return re.sub(r'\s+', ' ', s).strip()
