CREATE EXTENSION IF NOT EXISTS VECTOR;

CREATE TABLE IF NOT EXISTS items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type TEXT NOT NULL,
    source_ref TEXT,
    title TEXT,
    raw_content TEXT,
    created_at TIMESTAMP DEFAULT now(),
    status TEXT DEFAULT 'pending',
    content_hash VARCHAR(64) unique
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id UUID REFERENCES items(id) ON DELETE CASCADE,
    chunk_index INT,
    content TEXT,
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id UUID REFERENCES items(id) ON DELETE CASCADE,
    job_type TEXT,
    status TEXT DEFAULT 'pending',
    attempts INT DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS digests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    period_start TIMESTAMP,
    period_end TIMESTAMP,
    content TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_item_id ON chunks(item_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMP DEFAULT now(),
    ended_at TIMESTAMP,
    status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_name TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT now(),
    compressed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS memory_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    source_session_id UUID REFERENCES sessions(id),
    occurred_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT now(),
    importance FLOAT DEFAULT 0.5,
    access_count INT DEFAULT 0,
    last_accessed_at TIMESTAMP,
    superseded_by UUID REFERENCES memory_entries(id),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS session_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    summary_kind TEXT NOT NULL,
    content TEXT NOT NULL,
    first_message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    last_message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    message_count INT NOT NULL,
    created_at TIMESTAMP DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT chk_session_summary_kind CHECK (
        summary_kind IN ('standard_compression', 'emergency_compression', 'session_close', 'crash_recovery')
    ),
    CONSTRAINT chk_session_summary_message_count CHECK (message_count >= 0),
    CONSTRAINT chk_session_summary_source_range CHECK (
        first_message_id IS NOT NULL AND last_message_id IS NOT NULL
    )
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_session_summary_kind') THEN
        ALTER TABLE session_summaries ADD CONSTRAINT chk_session_summary_kind CHECK (
            summary_kind IN ('standard_compression', 'emergency_compression', 'session_close', 'crash_recovery')
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_session_summary_message_count') THEN
        ALTER TABLE session_summaries ADD CONSTRAINT chk_session_summary_message_count CHECK (message_count >= 0);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_session_summary_source_range') THEN
        ALTER TABLE session_summaries ADD CONSTRAINT chk_session_summary_source_range CHECK (
            first_message_id IS NOT NULL AND last_message_id IS NOT NULL
        );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_session_summaries_session_created ON session_summaries(session_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS uq_session_compression_span
    ON session_summaries(session_id, summary_kind, first_message_id, last_message_id)
    WHERE summary_kind IN ('standard_compression', 'emergency_compression');
CREATE UNIQUE INDEX IF NOT EXISTS uq_session_final_summary
    ON session_summaries(session_id)
    WHERE summary_kind IN ('session_close', 'crash_recovery');
CREATE INDEX IF NOT EXISTS idx_memory_type_date ON memory_entries(memory_type, occurred_at);
CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory_entries USING hnsw (embedding vector_cosine_ops);