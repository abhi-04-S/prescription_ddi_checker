-- =============================================================================
-- Migration 002: Drug Interactions RAG Table (Vector Store)
-- =============================================================================
-- Purpose: Store drug-drug interaction records with semantic embeddings for
--          hybrid retrieval (exact metadata match + cosine vector similarity).
--
-- Each row represents one known interaction between two generic salts,
-- containing:
--   • The two drug names (drug_a, drug_b) for exact metadata filtering
--   • A severity level (Critical, Major, Moderate)
--   • A clinical description of the interaction
--   • A natural-language semantic chunk (used to generate the embedding)
--   • A 384-dimensional vector embedding (from all-MiniLM-L6-v2)
--
-- The hybrid search RPC (Migration 003) queries this table using BOTH
-- exact drug name matching AND cosine similarity against the embedding.
-- =============================================================================

-- Enable the pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- The drug interactions vector store
CREATE TABLE IF NOT EXISTS drug_interactions_rag (
    id                      BIGSERIAL PRIMARY KEY,

    -- The two interacting generic salt names (alphabetically sorted on insert)
    -- e.g., drug_a = "Amlodipine", drug_b = "Telmisartan"
    drug_a                  TEXT NOT NULL,
    drug_b                  TEXT NOT NULL,

    -- Interaction severity: Critical > Major > Moderate
    -- "Critical" = life-threatening, contraindicated
    -- "Major"    = significant risk, monitor closely
    -- "Moderate" = mild risk, usually safe with supervision
    severity                TEXT NOT NULL CHECK (severity IN ('Critical', 'Major', 'Moderate')),

    -- Detailed clinical description of the interaction
    -- e.g., "Concurrent use increases risk of GI bleeding and renal impairment"
    clinical_description    TEXT NOT NULL,

    -- The full natural-language chunk used to generate the embedding
    -- Format: "Interaction between {drug_a} and {drug_b} (Severity: {severity}): {description}"
    semantic_chunk          TEXT NOT NULL,

    -- 384-dimensional embedding vector from all-MiniLM-L6-v2
    embedding               vector(384) NOT NULL,

    -- Source of the interaction data (for auditing/traceability)
    source                  TEXT DEFAULT 'manual',

    -- Timestamps
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Indexes ──

-- Composite index on drug pair for fast exact-match lookups
-- Covers both (A,B) and (B,A) ordering via the RPC function's WHERE clause
CREATE INDEX IF NOT EXISTS idx_interactions_drug_pair
    ON drug_interactions_rag (LOWER(drug_a), LOWER(drug_b));

-- Reverse pair index (for B,A lookups without OR overhead)
CREATE INDEX IF NOT EXISTS idx_interactions_drug_pair_rev
    ON drug_interactions_rag (LOWER(drug_b), LOWER(drug_a));

-- HNSW index on the embedding vector for fast approximate nearest neighbor search
-- Using cosine distance operator (<=>)
-- ef_construction=128 and m=16 are good defaults for ~10K-100K rows
CREATE INDEX IF NOT EXISTS idx_interactions_embedding_hnsw
    ON drug_interactions_rag
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- Index on severity for filtering
CREATE INDEX IF NOT EXISTS idx_interactions_severity
    ON drug_interactions_rag (severity);

-- Unique constraint: prevent duplicate interaction entries for the same drug pair
-- (drug_a is always alphabetically <= drug_b, enforced at insert time by the ingestion script)
CREATE UNIQUE INDEX IF NOT EXISTS idx_interactions_unique_pair
    ON drug_interactions_rag (LOWER(drug_a), LOWER(drug_b), severity);

