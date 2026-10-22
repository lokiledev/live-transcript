# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Clean Whisper text before it is stored."""

import re

# Phrases Whisper emits on silence, in English and French meetings.
_HALLUCINATIONS = {
    "thanks for watching",
    "thank you for watching",
    "thanks for listening",
    "please subscribe",
    "sous-titres réalisés par la communauté d'amara.org",
    "sous-titres par la communauté d'amara.org",
    "sous-titres par l'amara.org",
    "merci d'avoir regardé",
    "merci de votre attention",
}

_REPEAT = re.compile(
    r"(\b\w[\w']{1,40}(?:\s+\w[\w']{1,40}){0,6})(?:\s+\1){2,}",
    flags=re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"\w", flags=re.UNICODE)


def collapse_repeats(text: str) -> str:
    """Collapse a clause that was repeated three or more times."""
    return _REPEAT.sub(r"\1", text)


def usable_transcript(text: str) -> str | None:
    """Return cleaned speech, or None when the text is empty or a known hallucination."""
    cleaned = _WHITESPACE.sub(" ", collapse_repeats(text)).strip()
    if len(cleaned) < 2 or _WORD.search(cleaned) is None:
        return None
    key = cleaned.casefold().strip(" .!?,;:…")
    if key in _HALLUCINATIONS:
        return None
    return cleaned
