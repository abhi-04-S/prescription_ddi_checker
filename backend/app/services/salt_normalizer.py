"""
Salt Normalizer Service — Brand-to-Salt Resolution for Indian Prescriptions.

Takes raw prescription strings like:
    "Tab Dolo 650 1-0-1"
    "Cap Telmikind-AM OD"
    "Syp Augmentin 625 Duo BD x 5 days"

And resolves them into:
    ResolvedMedicine(brand_name="Dolo 650 Tablet", active_salts=["Paracetamol"], ...)
    ResolvedMedicine(brand_name="Telmikind-AM Tablet", active_salts=["Telmisartan", "Amlodipine"], ...)

Resolution hierarchy:
    1. Exact / prefix match on Supabase (ILIKE on brand_name) — fast, indexed
    2. Fuzzy match via word_similarity RPC (search_brand_fuzzy) — catches OCR typos
    3. If neither matches → returns "unresolved" so user/frontend can manually enter

Also detects:
    - Duplicate Salt Overdose Risk: two different brands (or multiple lines of same brand)
      resolving to the same active salt (e.g., Dolo 650 + Calpol 500, or 2x Dolo 650)
"""

import re
from supabase import Client

from app.models.responses import ResolvedMedicine


# ═══════════════════════════════════════════════════════════════════════════════
# 1. REGEX PATTERNS — Strip Indian prescription noise
# ═══════════════════════════════════════════════════════════════════════════════

# Dosage form prefixes (Tab, Cap, Syp, Inj, etc.)
DOSAGE_FORM_PATTERN = re.compile(
    r"^\s*(Tab\.?|Tablet|Cap\.?|Capsule|Syp\.?|Syrup|Inj\.?|Injection|"
    r"Susp\.?|Suspension|Drops?|Cream|Oint\.?|Ointment|Gel|Lotion|"
    r"Sachet|Powder|Respule|Inhaler|Patch)\s+",
    re.IGNORECASE
)

# Trailing treatment duration (x 5 days, for 7 days, x 2 weeks, etc.)
DURATION_PATTERN = re.compile(
    r"\s*(?:(?:x|for)\s*\d+\s*(?:days?|weeks?|months?)|(?:\d+)\s*(?:days?|weeks?|months?))\s*$",
    re.IGNORECASE
)

# Indian frequency/schedule codes (1-0-1, OD, BD, TDS, etc.)
FREQUENCY_PATTERN = re.compile(
    r"\s+("
    r"\d[\s-]\d[\s-]\d"           # 1-0-1, 1-1-1, 0-0-1, 1 0 1
    r"|OD|BD|TDS|QID|QDS"         # Once/Twice/Thrice/Four-times daily
    r"|SOS|PRN|STAT"              # As needed / immediately
    r"|HS|ON|AN|BBF"              # At bedtime / night / afternoon / before breakfast
    r"|AC|PC"                     # Before/After food
    r"|Q\d+H"                    # Every N hours (Q8H, Q12H)
    r"|ONCE\s+DAILY|TWICE\s+DAILY"
    r"|WEEKLY|MONTHLY"
    r"|AFTER\s+FOOD|BEFORE\s+FOOD|WITH\s+FOOD|EMPTY\s+STOMACH"
    r")\s*$",
    re.IGNORECASE
)

# Strength/dosage (650mg, 500 mg, 40/5 mg, 625, 75/20, etc.)
# Uses word boundary so digits inside OCR typos (e.g. '1' in 'Do1o') are not misidentified as strength
STRENGTH_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?)\s*(mg|g|ml|mcg|iu|%|units?)?\b",
    re.IGNORECASE
)

# Trailing noise (numbers that look like quantity: "x 10", "x 15 tablets", "#30")
QUANTITY_PATTERN = re.compile(
    r"\s*[x#]\s*\d+\s*(tablets?|caps?|capsules?|strips?)?\s*$",
    re.IGNORECASE
)

# True dosage forms at the END of the string (combo suffixes like Duo, Forte, Plus, DS are preserved)
TRAILING_FORM_PATTERN = re.compile(
    r"\s+(Tablet|Capsule|Syrup|Injection|Suspension|Drops?|Cream|"
    r"Ointment|Gel|Lotion|Sachet|Respule|Inhaler|Patch)\s*$",
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TEXT CLEANING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_prescription_text(raw_text: str) -> tuple[str, str | None]:
    """
    Cleans a raw prescription line by stripping dosage forms, duration,
    frequencies, quantities, and isolating the brand name + optional strength.

    Args:
        raw_text: e.g., "Tab Augmentin 625 BD x 5 days"

    Returns:
        (cleaned_brand_name, extracted_strength)
        e.g., ("Augmentin 625", "625")
    """
    if not raw_text or not raw_text.strip():
        return ("", None)

    text = raw_text.strip()

    # Remove leading dosage form prefix
    text = DOSAGE_FORM_PATTERN.sub("", text).strip()

    # Remove trailing treatment duration BEFORE frequency codes
    text = DURATION_PATTERN.sub("", text).strip()

    # Remove trailing frequency/schedule codes
    for _ in range(3):
        prev = text
        text = FREQUENCY_PATTERN.sub("", text).strip()
        if text == prev:
            break

    # Remove trailing quantity (x 10, #30, etc.)
    text = QUANTITY_PATTERN.sub("", text).strip()

    # Extract strength before removing it
    strength_match = STRENGTH_PATTERN.search(text)
    extracted_strength = None
    if strength_match:
        num_part = strength_match.group(1)
        unit_part = strength_match.group(2) or ""
        extracted_strength = f"{num_part}{unit_part}".strip()

    # Brand query keeps full text including strength and combo suffixes (e.g. "Duo")
    brand_query = re.sub(r"\s+", " ", text).strip()

    return (brand_query, extracted_strength)


def extract_brand_core(brand_query: str) -> str:
    """
    Extracts the core brand name by stripping trailing dosage forms and standalone numbers.
    Used for fuzzy matching when exact match fails.
    Preserves combination modifiers like 'Duo', 'Forte', 'Plus', 'DS'.

    "Dolo 650 Tablet" → "Dolo"
    "Telma 40 Tablet" → "Telma"
    "Augmentin 625 Duo" → "Augmentin Duo"
    """
    text = brand_query

    # Remove trailing true dosage forms (e.g. Tablet, Capsule)
    text = TRAILING_FORM_PATTERN.sub("", text).strip()

    # Remove standalone numbers (strengths like 650, 500, 40)
    text = re.sub(r"\b\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*(mg|g|ml|mcg|iu)?\b", "", text, flags=re.IGNORECASE).strip()

    # Clean up multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else brand_query


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SUPABASE BRAND LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════

def lookup_brand_exact(supabase: Client, brand_query: str) -> ResolvedMedicine | None:
    """
    Exact/prefix search on indian_medicines_catalog.
    Queries prefix match first (brand_query%), selects candidate with shortest brand_name
    (or exact case-insensitive match). Falls back to %brand_query% only if no prefix match.
    """
    # 1. Try prefix match first
    result = (
        supabase.table("indian_medicines_catalog")
        .select("brand_name, active_salts, strength, manufacturer")
        .ilike("brand_name", f"{brand_query}%")
        .limit(5)
        .execute()
    )
    rows = result.data if result.data else []

    # 2. Fall back to %brand_query% if prefix match returns no rows
    if not rows:
        result = (
            supabase.table("indian_medicines_catalog")
            .select("brand_name, active_salts, strength, manufacturer")
            .ilike("brand_name", f"%{brand_query}%")
            .limit(5)
            .execute()
        )
        rows = result.data if result.data else []

    if not rows:
        return None

    # Check for exact case-insensitive match first
    query_lower = brand_query.strip().lower()
    exact = next((r for r in rows if r["brand_name"].strip().lower() == query_lower), None)
    chosen = exact if exact else min(rows, key=lambda r: len(r["brand_name"]))

    return ResolvedMedicine(
        original_text=brand_query,
        brand_name=chosen["brand_name"],
        active_salts=chosen["active_salts"] if isinstance(chosen["active_salts"], list) else [],
        strength=chosen.get("strength"),
        match_method="db_exact",
        match_score=1.0,
    )


def lookup_brand_fuzzy(
    supabase: Client,
    brand_query: str,
    similarity_cutoff: float = 0.55,
) -> ResolvedMedicine | None:
    """
    Fuzzy search via search_brand_fuzzy RPC (powered by word_similarity).
    Catches OCR typos like "Do1o" → "Dolo", "Telmiknd" → "Telmikind".
    """
    core_name = extract_brand_core(brand_query)

    result = supabase.rpc(
        "search_brand_fuzzy",
        {
            "query_brand": core_name,
            "similarity_cutoff": similarity_cutoff,
            "max_results": 1,
        },
    ).execute()

    if result.data:
        row = result.data[0]
        return ResolvedMedicine(
            original_text=brand_query,
            brand_name=row["brand_name"],
            active_salts=row["active_salts"] if isinstance(row["active_salts"], list) else [],
            strength=row.get("strength"),
            match_method="db_fuzzy",
            match_score=row.get("match_score"),
        )

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN NORMALIZATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_single_medicine(
    raw_text: str,
    supabase: Client,
    similarity_cutoff: float = 0.55,
) -> ResolvedMedicine:
    """
    Full normalization pipeline for a single prescription line.

    1. Clean text (strip Tab/Cap, duration, frequencies)
    2. Try exact/prefix DB match on brand_query
    3. Try exact/prefix DB match on core_name
    4. Try fuzzy DB match via word_similarity RPC
    5. Return unresolved if nothing matches
    """
    brand_query, extracted_strength = clean_prescription_text(raw_text)

    if not brand_query:
        return ResolvedMedicine(
            original_text=raw_text,
            match_method="unresolved",
        )

    # Step 1: Exact / prefix match
    result = lookup_brand_exact(supabase, brand_query)
    if result:
        result.original_text = raw_text
        return result

    # Step 2: Try with core brand name (no standalone strength/dosage forms)
    core_name = extract_brand_core(brand_query)
    if core_name != brand_query:
        result = lookup_brand_exact(supabase, core_name)
        if result:
            result.original_text = raw_text
            result.match_method = "db_exact_core"
            return result

    # Step 3: Fuzzy match (handles OCR character mistakes)
    result = lookup_brand_fuzzy(supabase, brand_query, similarity_cutoff)
    if result:
        result.original_text = raw_text
        return result

    # Step 4: Unresolved
    return ResolvedMedicine(
        original_text=raw_text,
        brand_name=brand_query,
        match_method="unresolved",
    )


def normalize_prescription(
    medicine_lines: list[str],
    supabase: Client,
    similarity_cutoff: float = 0.55,
) -> dict:
    """
    Normalizes a full prescription (multiple medicine lines).

    Returns:
        {
            "resolved_medicines": [ResolvedMedicine, ...],
            "unique_salts": ["Paracetamol", "Ibuprofen", ...],
            "duplicate_salt_warnings": ["Paracetamol (found in: ...)"],
            "unresolved_medicines": ["some unknown brand"],
            "salt_sources": {"Paracetamol": ["Dolo 650", "Calpol 500"]}
        }
    """
    resolved: list[ResolvedMedicine] = []
    unresolved: list[str] = []

    # Track which salts appear and from which brands (appends every occurrence)
    salt_to_brands: dict[str, list[str]] = {}

    for line in medicine_lines:
        med = normalize_single_medicine(line, supabase, similarity_cutoff)
        resolved.append(med)

        if med.match_method == "unresolved":
            unresolved.append(line)
        else:
            brand_label = med.brand_name or med.original_text
            for salt in med.active_salts:
                salt_lower = salt.lower()
                if salt_lower not in salt_to_brands:
                    salt_to_brands[salt_lower] = []
                # Append every occurrence to track same-brand duplicate prescriptions
                salt_to_brands[salt_lower].append(brand_label)

    # Collect unique salts preserving original casing
    unique_salts: list[str] = []
    seen_salts: set[str] = set()
    for med in resolved:
        for salt in med.active_salts:
            if salt.lower() not in seen_salts:
                seen_salts.add(salt.lower())
                unique_salts.append(salt)

    # Map each salt to its source brands
    salt_sources: dict[str, list[str]] = {}
    for salt in unique_salts:
        salt_sources[salt] = salt_to_brands.get(salt.lower(), [])

    # Detect duplicate salt overdose risk (salt appeared 2+ times across brands/lines)
    duplicate_salt_warnings: list[str] = []
    for salt in unique_salts:
        brands = salt_sources[salt]
        if len(brands) >= 2:
            warning = f"{salt} (found in: {', '.join(brands)})"
            duplicate_salt_warnings.append(warning)

    return {
        "resolved_medicines": resolved,
        "unique_salts": unique_salts,
        "duplicate_salt_warnings": duplicate_salt_warnings,
        "unresolved_medicines": unresolved,
        "salt_sources": salt_sources,
    }
