CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id  SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    source_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    embedding VECTOR(384),
    UNIQUE (document_id, chunk_index, embed_model)
);

-- Fast approximate nearest-neighbour search on cosine distance.
CREATE INDEX chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Every retrieval filters on this column, so give it an index too.
CREATE INDEX chunks_embed_model_idx ON chunks (embed_model);

CREATE TABLE runs (
  id          SERIAL PRIMARY KEY,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  question    TEXT NOT NULL,
  answer      TEXT,
  llm_model   TEXT,
  embed_model TEXT,
  steps       JSONB,
  metrics     JSONB,
  latency_ms  INT,
  tokens_in   INT,
  tokens_out  INT
);

CREATE TABLE eval_results (
  id           SERIAL PRIMARY KEY,
  eval_label   TEXT,
  question_id  TEXT,
  hit          BOOLEAN,
  faithfulness REAL,
  correct      BOOLEAN,
  run_id       INT REFERENCES runs(id)
);