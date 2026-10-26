from src.chunking import chunk_text, clean_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", max_words=10) == []
    assert chunk_text("   \n\n  ", max_words=10) == []


def test_short_text_is_one_chunk():
    text = "RAG retrieves text before it answers."
    assert chunk_text(text, max_words=20, overlap_words=5) == [text]


def test_chunks_respect_max_words():
    sentences = [f"Sentence number {i} talks about retrieval." for i in range(1, 31)]
    chunks = chunk_text(" ".join(sentences), max_words=20, overlap_words=4)
    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 20 for chunk in chunks)


def test_consecutive_chunks_overlap():
    sentences = [f"Word{i} pair." for i in range(1, 41)]
    chunks = chunk_text(" ".join(sentences), max_words=12, overlap_words=4)
    assert len(chunks) >= 2
    assert chunks[0].split()[-4:] == chunks[1].split()[:4]


def test_long_unpunctuated_text_is_hard_split():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = chunk_text(text, max_words=20, overlap_words=0)
    assert all(len(chunk.split()) <= 20 for chunk in chunks)
    assert sum(len(chunk.split()) for chunk in chunks) == 50


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\r\n\r\n\r\nc") == "a b\n\nc"


def test_clean_text_strips_markdown_heading_markers():
    assert clean_text("## Start with a gold set\n\nA gold set is small.") == (
        "Start with a gold set\n\nA gold set is small."
    )
