CREATE TABLE IF NOT EXISTS nextgenrag_questions (
    domain_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    query TEXT NOT NULL,
    gpt_adherence INTEGER,
    gpt_relevance REAL,
    gpt_utilization REAL,
    gpt_completeness REAL,
    claude_adherence INTEGER,
    claude_relevance REAL,
    claude_utilization REAL,
    claude_completeness REAL
);

CREATE TABLE IF NOT EXISTS nextgenrag_sample_questions (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nextgenrag_v1 (
    session_id TEXT NOT NULL,
    id INTEGER NOT NULL,
    chunk_size INTEGER,
    chunk_overlap INTEGER,
    vector_db TEXT,
    retreival_type TEXT,
    gen_model TEXT,
    embed_model TEXT,
    eval_model TEXT,
    search_type TEXT,
    search_kwargs TEXT,
    response TEXT,
    context TEXT,
    adherence INTEGER,
    relevance REAL,
    utilization REAL,
    completeness REAL,
    parse_error INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (id) REFERENCES nextgenrag_sample_questions(id)
);

CREATE INDEX IF NOT EXISTS idx_nextgenrag_v1_session ON nextgenrag_v1(session_id);
CREATE INDEX IF NOT EXISTS idx_nextgenrag_v1_id ON nextgenrag_v1(id);
