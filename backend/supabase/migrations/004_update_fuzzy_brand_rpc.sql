-- =============================================================================
-- Migration 004: Upgrade Fuzzy Brand Search to word_similarity
-- =============================================================================
-- Why: `similarity()` compares entire strings, so short queries like "Dolo"
--      score poorly against "Dolo 650 Tablet" (lots of unmatched characters).
--      `word_similarity()` finds the best-matching substring window, so
--      "Dolo" scores high against "Dolo 650 Tablet" because it matches
--      the "Dolo" word perfectly.
--
-- Also raises default cutoff from 0.3 to 0.55 to reduce false positives
-- against the full 10K+ catalog while still catching OCR typos.
-- =============================================================================

-- Drop old function
DROP FUNCTION IF EXISTS search_brand_fuzzy(TEXT, FLOAT, INT);

CREATE OR REPLACE FUNCTION search_brand_fuzzy(
    query_brand TEXT,
    similarity_cutoff FLOAT DEFAULT 0.55,
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
        word_similarity(query_brand, m.brand_name) AS match_score
    FROM indian_medicines_catalog m
    WHERE word_similarity(query_brand, m.brand_name) > similarity_cutoff
    ORDER BY match_score DESC, length(m.brand_name) ASC
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;
