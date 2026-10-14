-- Safe to run many times: every statement is IF NOT EXISTS.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    id               text PRIMARY KEY,               -- OpenAlex id, e.g. W2741809807
    doi              text,
    title            text NOT NULL,
    abstract         text NOT NULL,
    year             int,
    publication_date date,
    venue            text,
    cited_by_count   int NOT NULL DEFAULT 0,
    authors          jsonb NOT NULL DEFAULT '[]',
    referenced_ids   text[] NOT NULL DEFAULT '{}',
    arxiv_id         text,
    fulltext         text CHECK (fulltext IN ('pdf', 'abstract')),  -- set once the paper is ingested
    fetched_at       timestamptz NOT NULL DEFAULT now(),
    embedding        vector(384) NOT NULL,
    tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || abstract)) STORED
);
CREATE INDEX IF NOT EXISTS papers_embedding_hnsw ON papers USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS papers_tsv_gin ON papers USING gin (tsv);
CREATE INDEX IF NOT EXISTS papers_year ON papers (year);

CREATE TABLE IF NOT EXISTS chunks (
    id        bigserial PRIMARY KEY,
    paper_id  text NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    ord       int NOT NULL,
    section   text NOT NULL,
    text      text NOT NULL,
    tokens    int NOT NULL,
    embedding vector(384) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (paper_id, ord)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS jobs (
    id          uuid PRIMARY KEY,
    topic       text NOT NULL,
    settings    jsonb NOT NULL,
    status      text NOT NULL CHECK (status IN ('queued', 'running', 'waiting_approval', 'succeeded', 'failed', 'canceled')),
    attempts    int NOT NULL DEFAULT 0,
    lease_until timestamptz,
    worker_id   text,
    resume      jsonb,                             -- approval answer waiting to be passed to the graph
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);

CREATE TABLE IF NOT EXISTS job_papers (
    job_id          uuid NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    paper_id        text NOT NULL REFERENCES papers (id),
    signals         jsonb NOT NULL,                -- bm25, dense, facet_rel per facet; ppr
    ranks           jsonb NOT NULL,
    fused           real NOT NULL,
    fused_rank      int NOT NULL,
    verdict         jsonb,
    posterior       real,
    selected        boolean NOT NULL DEFAULT false,
    selection_order int,
    PRIMARY KEY (job_id, paper_id)
);

CREATE TABLE IF NOT EXISTS sections (
    job_id       uuid NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    idx          int NOT NULL,
    title        text NOT NULL,
    goal         text NOT NULL,
    paper_ids    text[] NOT NULL,
    draft        text,
    final        text,
    status       text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done', 'dropped')),
    verification jsonb NOT NULL DEFAULT '{}',
    PRIMARY KEY (job_id, idx)
);

CREATE TABLE IF NOT EXISTS evidence (
    job_id      uuid NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    card_id     text NOT NULL,
    section_idx int NOT NULL,
    paper_id    text NOT NULL REFERENCES papers (id),
    chunk_id    bigint NOT NULL REFERENCES chunks (id),
    claim       text NOT NULL,
    quote       text NOT NULL,
    PRIMARY KEY (job_id, card_id)
);

CREATE TABLE IF NOT EXISTS outputs (
    job_id         uuid PRIMARY KEY REFERENCES jobs (id) ON DELETE CASCADE,
    paper_md       text NOT NULL,
    references_bib text NOT NULL,
    report         jsonb NOT NULL,
    report_md      text NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                bigserial PRIMARY KEY,
    job_id            uuid NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    node              text NOT NULL,
    model             text NOT NULL,
    input_tokens      int NOT NULL,
    output_tokens     int NOT NULL,
    cache_read_tokens int NOT NULL,
    cost_usd          numeric(12, 6) NOT NULL,
    latency_ms        int NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS llm_calls_job ON llm_calls (job_id);

CREATE TABLE IF NOT EXISTS job_events (
    id         bigserial PRIMARY KEY,
    job_id     uuid NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    node       text NOT NULL,
    event      text NOT NULL,
    data       jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS job_events_job ON job_events (job_id, id);
