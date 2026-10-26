# Common RAG Failure Modes

## Retrieval miss

The answer exists in the corpus, but it is not in the top-k chunks. Causes
include a query that uses different vocabulary from the document, chunks that
are too large to be specific, an embedding model that does not represent the
domain well, and a top-k that is simply too small.

This is the most common failure, and the easiest to misdiagnose. The model gets
blamed for an answer it never had the material to produce. Hit rate at k is the
metric that catches it.

## Bad chunking

Chunking fails quietly. A table split across two chunks becomes two sets of
numbers with no headers. A definition split at a sentence boundary becomes two
half-definitions. A chunk that starts mid-sentence embeds poorly because its
first words are pronouns with no referent.

Symptoms include retrieval that returns almost-right chunks from the right
document, and answers that are vague in a way the source is not. Overlap,
sentence-aware splitting, and smaller chunks are the usual fixes.

## Lost in the middle

Models attend more strongly to the beginning and the end of a long context than
to the middle. If the useful chunk sits in position five of eight, it can be
effectively ignored even though it is present.

Keep top-k small, put the highest-scoring chunk first, and do not pad the
context with barely relevant text to look thorough.

## Hallucination despite context

The model adds outside knowledge, overstates a claim, or merges two sources
into a statement neither one supports. A strict instruction to use only the
sources reduces this but does not eliminate it.

Mitigations are a strict prompt, an explicit refusal sentence, a faithfulness
check, and a visible source panel so a reader can verify the claim in seconds.

## Citation drift

The answer cites a source that does not contain the claim. Small models are
unreliable at exact numbering, especially when two chunks say similar things.
The citation looks authoritative, which makes this failure worse than no
citation at all.

Check citations after generation instead of trusting them. Highlighting the
sentences in a chunk that are closest to each answer sentence makes drift
visible.

## Stale index and embedding mismatch

Changing the embedding model without re-embedding leaves old vectors in the
table. The new query vector and the old document vectors live in different
spaces, so similarity scores become meaningless while the system keeps
returning results.

Changing chunk size without re-ingesting has a milder version of the same
problem: the index no longer matches the configuration you think you are
running.

## Query-document mismatch

The question says car, the document says vehicle. The question is a full
sentence, the document is a table row. Embeddings bridge some of this, but not
all of it.

Query rewriting, adding keywords, or embedding a hypothetical answer instead of
the question are common fixes. Each one is an experiment to measure, not a
default to assume.

## A short mitigation table

Retrieval miss: raise k, improve chunking, try another embedding model.
Bad chunking: sentence-aware splitting, overlap, test chunk sizes.
Lost in the middle: smaller k, best chunk first.
Hallucination: strict prompt, refusal rule, faithfulness check.
Citation drift: post-hoc verification and highlighting.
Stale index: re-ingest after every embedding or chunking change.
Query mismatch: rewrite the query, then measure hit rate again.
