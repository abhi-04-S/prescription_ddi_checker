-- =============================================================================
-- Migration 001: Indian Medicines Catalog
-- =============================================================================
-- Purpose: Store Indian commercial brand names and their underlying generic
--          chemical salts. This is the lookup table that powers brand → salt
--          resolution (e.g., "Dolo 650" → ["Paracetamol"]).
--
-- Key features:
--   • pg_trgm extension for fuzzy brand name matching (handles OCR typos)
--   • JSONB array for active_salts (supports multi-salt combination tablets)
--   • GIN indexes for fast trigram similarity + JSONB containment queries
-- =============================================================================

-- Enable the trigram extension for fuzzy text matching
-- (lets us do: SELECT * WHERE similarity(brand_name, 'Dolo') > 0.7)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- The main medicines lookup table
CREATE TABLE IF NOT EXISTS indian_medicines_catalog (
    id              BIGSERIAL PRIMARY KEY,

    -- The commercial brand name as printed on the strip/box
    -- e.g., "Dolo", "Telmikind-AM", "Ecosprin Gold"
    brand_name      TEXT NOT NULL,

    -- Array of generic chemical salt names (INN/USAN names)
    -- Stored as JSONB array for flexibility with combination tablets
    -- e.g., ["Paracetamol"] or ["Telmisartan", "Amlodipine"]
    active_salts    JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Strength/dosage printed on the pack (e.g., "650mg", "40mg/5mg")
    strength        TEXT,

    -- Pharmaceutical company (e.g., "Micro Labs", "Mankind Pharma")
    manufacturer    TEXT,

    -- Timestamps for auditing
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Indexes ──

-- Trigram index on brand_name: enables fast fuzzy matching
-- Used when OCR misreads a name (e.g., "Do1o" should still match "Dolo")
CREATE INDEX IF NOT EXISTS idx_medicines_brand_trgm
    ON indian_medicines_catalog
    USING GIN (brand_name gin_trgm_ops);

-- Standard B-tree index for exact lookups (faster than trigram for exact match)
CREATE INDEX IF NOT EXISTS idx_medicines_brand_exact
    ON indian_medicines_catalog (LOWER(brand_name));

-- GIN index on active_salts JSONB for containment queries
-- e.g., WHERE active_salts @> '["Paracetamol"]'
CREATE INDEX IF NOT EXISTS idx_medicines_salts_gin
    ON indian_medicines_catalog
    USING GIN (active_salts);

-- ── Helper function: fuzzy brand search ──
-- Called from salt_normalizer.py when exact CSV lookup fails

CREATE OR REPLACE FUNCTION search_brand_fuzzy(
    query_brand TEXT,
    similarity_cutoff FLOAT DEFAULT 0.3,
    max_results INT DEFAULT 5
)
RETURNS TABLE (
    id              BIGINT,
    brand_name      TEXT,
    active_salts    JSONB,
    strength        TEXT,
    manufacturer    TEXT,
    match_score     REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.brand_name,
        m.active_salts,
        m.strength,
        m.manufacturer,
        similarity(m.brand_name, query_brand) AS match_score
    FROM indian_medicines_catalog m
    WHERE similarity(m.brand_name, query_brand) > similarity_cutoff
    ORDER BY match_score DESC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

