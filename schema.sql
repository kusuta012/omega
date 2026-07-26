CREATE EXTENSION IF NOT EXISTS VECTOR;

CREATE TABLE items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type TEXT NOT NULL,
    source_ref TEXT,
    title TEXT,
    raw_content TEXT,
    created_at TIMESTAMP DEFAULT now(),
    status TEXT DEFAULT 'pending',
    content_hash VARCHAR(64) unique
);

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id UUID REFERENCES items(id) ON DELETE CASCADE,
    chunk_index INT,
    content TEXT,
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id UUID REFERENCES items(id),
    job_type TEXT,
    status TEXT DEFAULT 'pending',
    attempts INT DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE digests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    period_start TIMESTAMP,
    period_end TIMESTAMP,
    content TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX idx_chunks_item_id ON chunks(item_id);
CREATE INDEX idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);