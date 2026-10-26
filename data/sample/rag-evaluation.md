# Evaluating a RAG System

## Start with a gold set

A gold set is a small collection of questions with known answers. Thirty to
fifty questions is enough to see a difference between configurations. For each
question record the expected answer, the document that contains it, and
whether the corpus can answer it at all.

Include unanswerable questions. Ten to twenty percent of the set should have no
answer in the corpus, so you can measure whether the system refuses instead of
inventing something.

Write the first ten questions before building the retriever. They shape the
design: if every question depends on a table that chunking splits in half, you
learn that early.

## Retrieval metrics

Hit rate at k is the fraction of questions for which the correct source appears
anywhere in the top-k results. It ignores position: a correct chunk at rank one
and a correct chunk at rank four both count as a hit.

Mean reciprocal rank, MRR, fixes that blind spot. For each question, take the
rank of the first correct chunk and compute one divided by that rank. A correct
chunk at rank one scores 1.0, at rank two 0.5, at rank four 0.25. Average over
all questions.

Recall at k asks what fraction of all relevant chunks were retrieved. Precision
at k asks what fraction of the returned chunks were relevant. With one gold
chunk per question, hit rate and recall at k are the same number.

## Generation metrics

Faithfulness asks whether every claim in the answer is supported by the
retrieved context. It is judged claim by claim, so a correct answer with one
invented detail scores lower than a shorter answer that stays inside the
sources.

Answer relevance asks whether the answer addresses the question that was asked.
An answer can be faithful and irrelevant at the same time: it faithfully
summarises a chunk that does not answer the question.

Correctness compares the answer with the gold answer. A human can do this, or a
second model can, but the gold answer has to exist first.

## Behaviour metrics

Refusal accuracy measures two things at once: the system says I do not know
when the corpus lacks the answer, and it does not refuse when the answer is
present. Report both error directions; a system that always refuses would
otherwise look perfect on the first half.

Latency and token counts belong in every report. An answer that is slightly
better but ten times slower is a different product decision, not a free win.

## LLM as judge

A judge is a second model call with a rubric and a request for JSON. Set
temperature to zero and ask for one verdict per claim, so a single bad sentence
does not hide inside a paragraph.

Small judges are noisy. Spot-check ten to twenty verdicts by hand before you
trust a metric, and use the largest model your machine can run for judging.
Treat a faithfulness score as a rough instrument, not a measurement with two
decimal places of meaning.

## Record the configuration

Every result needs its configuration beside it: chat model, embedding model,
chunk size, overlap, top-k, and whether query rewriting was on. A number
without its configuration is not reproducible, and an experiment you cannot
reproduce is an anecdote.
