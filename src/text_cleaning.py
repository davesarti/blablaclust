"""Text cleaning utilities for Amazon Review data points."""

import re
import unicodedata


def clean_text(title: str, text: str) -> str:
    """Combine title and text into a single clean string for embedding."""
    combined = title.strip() + " " + text.strip()
    combined = _normalize_unicode(combined)
    combined = _fix_double_quotes(combined)
    combined = _collapse_repeated_chars(combined)
    combined = _collapse_whitespace(combined)
    return combined


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
