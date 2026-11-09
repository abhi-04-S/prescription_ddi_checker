-- =============================================================================
-- Migration 003: Hybrid Search RPC Function — match_drug_interactions
-- =============================================================================
-- Purpose: Single RPC function that performs HYBRID retrieval:
--   1. FIRST tries exact metadata match (drug_a/drug_b names) — fast, precise
--   2. THEN falls back to cosine vector similarity — catches semantic matches
--      that exact text wouldn't find (synonyms, alternate phrasings)
--
-- Called from: rag_checker.py via Supabase client
-- Input:  drug pair names + 384-d query embedding
-- Output: matching interaction rows deduplicated and ranked by relevance
-- =============================================================================

-- Drop existing function if it exists (clean re-deploy)
DROP FUNCTION IF EXISTS match_drug_interactions(TEXT, TEXT, vector(384), FLOAT, INT);

CREATE OR REPLACE FUNCTION match_drug_interactions(
    query_drug_a         TEXT,
    query_drug_b         TEXT,
    query_embedding      vector(384),
    similarity_threshold FLOAT DEFAULT 0.82,
    match_count          INT DEFAULT 5
)
RETURNS TABLE (
    id                   BIGINT,
    drug_a               TEXT,
    drug_b               TEXT,
    severity             TEXT,
    clinical_description TEXT,
    semantic_chunk       TEXT,
    similarity_score     FLOAT,
    match_type           TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        deduped.id,
        deduped.drug_a,
        deduped.drug_b,
        deduped.severity,
        deduped.clinical_description,
        deduped.semantic_chunk,
        deduped.similarity_score,
        deduped.match_type
    FROM (
        -- Inner deduplication: if the same row is matched by both exact metadata
        -- and vector similarity, keep the 'exact' row ('exact' < 'semantic' alphabetically)
        SELECT DISTINCT ON (sub.id)
            sub.id,
            sub.drug_a,
            sub.drug_b,
            sub.severity,
            sub.clinical_description,
            sub.semantic_chunk,
            sub.similarity_score,
            sub.match_type
        FROM (
            -- Part 1: Exact metadata matches (fast — uses composite B-tree indexes)
            SELECT
                d.id,
                d.drug_a,
                d.drug_b,
                d.severity,
                d.clinical_description,
                d.semantic_chunk,
                1.0::FLOAT AS similarity_score,
                'exact'::TEXT AS match_type
            FROM drug_interactions_rag d
            WHERE
                (LOWER(d.drug_a) = LOWER(query_drug_a) AND LOWER(d.drug_b) = LOWER(query_drug_b))
                OR
                (LOWER(d.drug_a) = LOWER(query_drug_b) AND LOWER(d.drug_b) = LOWER(query_drug_a))

            UNION ALL

            -- Part 2: Semantic similarity matches (uses HNSW cosine vector index)
            SELECT
                d.id,
                d.drug_a,
                d.drug_b,
                d.severity,
                d.clinical_description,
                d.semantic_chunk,
                (1 - (d.embedding <=> query_embedding))::FLOAT AS similarity_score,
                'semantic'::TEXT AS match_type
            FROM drug_interactions_rag d
            WHERE (1 - (d.embedding <=> query_embedding)) >= similarity_threshold
        ) sub
        ORDER BY sub.id, sub.match_type ASC
    ) deduped
    -- Final ranking:
    -- 1. 'exact' matches before 'semantic' matches
    -- 2. Highest cosine similarity score first
    -- 3. Critical > Major > Moderate severity tie-breaker
    ORDER BY
        CASE WHEN deduped.match_type = 'exact' THEN 0 ELSE 1 END ASC,
        deduped.similarity_score DESC,
        CASE deduped.severity
            WHEN 'Critical' THEN 1
            WHEN 'Major'    THEN 2
            WHEN 'Moderate' THEN 3
            ELSE 4
        END ASC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;