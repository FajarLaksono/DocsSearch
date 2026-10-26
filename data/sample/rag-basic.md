# Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) is a pattern for answering questions with
a language model by giving it relevant text at answer time, instead of relying
on what the model memorised during training.

## Why RAG exists

A model's weights are frozen when training ends. Anything newer, private, or
local is invisible to it. RAG addresses that by searching a document
collection for passages related to the question and placing those passages in
the prompt.

The pattern buys three things. Freshness: the collection can be updated
without touching the model. Provenance: the answer can point at the passage it
used. Control: you decide what the model is allowed to see.

RAG does not make a weak model reason well, and it does not guarantee that an
answer is true. It only guarantees that the model was shown specific text.

## The pipeline

Ingestion runs ahead of time. Documents are loaded, cleaned, split into
chunks, embedded, and stored.

Retrieval runs per question. The question is embedded with the same model,
and the store returns the chunks whose vectors are closest.

Generation builds a prompt from the question and the retrieved chunks, and the
model writes an answer grounded in that context.

Ingestion and retrieval are separate on purpose. Embedding is expensive and
changes only when documents change. Retrieval has to be fast enough to run on
every keystroke if you want.

## Chunking

Documents are split into chunks because an embedding compresses text into one
fixed-length vector. A whole book in one vector is useless: the result points
at the average of everything and is close to nothing.

A common starting point is 300 to 500 tokens per chunk with 10 to 20 percent
overlap. Chunk size is a trade-off. Small chunks retrieve precisely but lose
the surrounding context. Large chunks keep context but dilute the embedding,
so the vector represents several ideas at once.

Overlap matters at a boundary. A definition that straddles two chunks appears
whole in neither one, so neither embedding captures it. Repeating the last
sentences of a chunk at the start of the next keeps that definition intact in
at least one chunk.

Chunk boundaries should follow paragraphs and sentences. Cutting in the middle
of a sentence produces chunks that begin with a pronoun and end with a comma;
both the embedding and the reader suffer.

## Embeddings and similarity

An embedding model turns text into a fixed-length vector, typically a few
hundred floats. Texts with similar meaning produce vectors that point in
similar directions, even when they share no words.

Cosine similarity measures the angle between two vectors. It ranges from -1 to
1 for arbitrary vectors and is usually between 0 and 1 for text embeddings.
pgvector's <=> operator returns cosine distance, so similarity is 1 minus
distance.

The embedding model used at ingest time must be the same one used at query
time. Vectors from different models are not comparable, even when the vector
size is identical. Re-embedding after a model change is mandatory, not
optional.

## What RAG does not solve

RAG cannot answer from a document that was never ingested. It cannot repair a
retrieval step that returns the wrong chunk. It cannot stop a model from
adding outside knowledge unless the prompt forbids it and something checks the
result. Evaluation exists because none of these failures announce themselves.
