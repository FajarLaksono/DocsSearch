"""Split documents into overlapping, sentence-aware chunks.

Token counts here are approximate on purpose: exact tokenisation needs the
model's own tokenizer, and for chunk sizing a 10-20% error is irrelevant.
"""

from __future__ import annotations

import re

# English text is roughly 1.3 tokens per word.
# so one token is about 0.75 words. Approximate, documented, deterministic.
WORDS_PER_TOKEN = 0.75

def token_target_to_words(tokens: int) -> int:
    return max(1, int(tokens * WORDS_PER_TOKEN))

def clean_text(text: str) -> str:
    """Normalise line endings, strip heading markers, collapse whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)  # "## Chunking" -> "Chunking"
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _units(text: str, max_words: int) -> list[str]:
    """Paragraphs -> sentences -> hard slices for over-long sentences."""
    units: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            words = sentence.split()
            if len(words) <= max_words:
                units.append(sentence)
            else:
                for start in range(0, len(words), max_words):
                    units.append(" ".join(words[start : start + max_words]))
    return units

def chunk_text(text: str, max_words: int, overlap_words: int = 0) -> list[str]:
    """Pack units into chunks of at most ``max_words`` words.

    The overlap is only kept when it fits inside the budget of the chunk that
    is about to start, so no chunk ever exceeds ``max_words``.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    overlap_words = max(0, min(overlap_words, max_words - 1))

    text = clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    current: list[str] = []
    size = 0

    for unit in _units(text, max_words):
        unit_words = len(unit.split())
        if current and size + unit_words > max_words:
            chunks.append(" ".join(current))

            # Reserve room for the incoming unit before deciding how much of
            # the previous chunk can be repeated.
            budget = max_words - unit_words
            keep = min(overlap_words, budget)
            tail: list[str] = []
            tail_size = 0
            for previous in reversed(current):
                previous_words = len(previous.split())
                if tail_size + previous_words > keep:
                    break
                tail.insert(0, previous)
                tail_size += previous_words

            current = tail
            size = tail_size

        current.append(unit)
        size += unit_words

    if current:
        chunks.append(" ".join(current))
    return chunks