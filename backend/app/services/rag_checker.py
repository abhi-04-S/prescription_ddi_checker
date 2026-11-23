"""
RAG Interaction Checker Service — Combinatorial Hybrid Vector Retrieval & Safety Audit.

Handles:
  1. Base-salt stripping (strips 'Hydrochloride', 'Sodium', 'Besylate', etc. for DB matching)
  2. Combinatorial pair generation (itertools.combinations)
  3. Exact mathematical Fixed-Dose Combination (FDC) guard rule
  4. INN / USAN bidirectional alias expansion (Paracetamol <-> Acetaminophen)
  5. Batched local vectorization via SentenceTransformer (all-MiniLM-L6-v2)
  6. Hybrid retrieval via Supabase match_drug_interactions RPC
  7. Strict Semantic False-Positive Guard (discards unrelated vector neighbors)
  8. Structured allergy cross-checking (alias-aware + clinical drug class aware)
"""

import re
import itertools
from typing import Any
from supabase import Client
from sentence_transformers import SentenceTransformer

from app.config import settings


# ═══════════════════════════════════════════════════════════════════════════════
# 1. COUNTER-ION & BASE SALT STRIPPING
# ═══════════════════════════════════════════════════════════════════════════════

SALT_SUFFIX_PATTERN = re.compile(
    r"\b("
    r"Hydrochloride|Hcl|Sodium|Potassium|Calcium|Maleate|Besylate|"
    r"Tartrate|Mesylate|Sulfate|Sulphate|Succinate|Phosphate|Nitrate|"
    r"Fumarate|Acetate|Citrate|Gluconate|Bromide|Iodide|"
    r"IP|BP|USP"
    r")\b",
    re.IGNORECASE
)


def get_base_salt(salt_name: str) -> str:
    """
    Strips pharmaceutical counter-ion suffixes so catalog salts match DrugBank.
    e.g., 'Metformin Hydrochloride' -> 'Metformin', 'Diclofenac Sodium' -> 'Diclofenac'
    """
    if not salt_name:
        return ""
    cleaned = SALT_SUFFIX_PATTERN.sub("", salt_name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else salt_name.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INN / USAN ALIASES & VALID PAIR MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

DRUG_ALIASES: dict[str, list[str]] = {
    "paracetamol": ["acetaminophen"],
    "acetaminophen": ["paracetamol"],
    "amoxicillin": ["amoxycillin"],
    "amoxycillin": ["amoxicillin"],
    "clavulanic acid": ["clavulanate", "potassium clavulanate"],
    "clavulanate": ["clavulanic acid"],
    "aspirin": ["acetylsalicylic acid", "asa"],
    "acetylsalicylic acid": ["aspirin"],
    "frusemide": ["furosemide"],
    "furosemide": ["frusemide"],
    "salbutamol": ["albuterol"],
    "albuterol": ["salbutamol"],
    "rifampicin": ["rifampin"],
    "rifampin": ["rifampicin"],
    "ibuprofen": ["ibuprofen"],
    "diclofenac": ["diclofenac"],
    "telmisartan": ["telmisartan"],
    "amlodipine": ["amlodipine"],
    "pantoprazole": ["pantoprazole"],
    "atorvastatin": ["atorvastatin"],
    "metformin": ["metformin"],
    "glimepiride": ["glimepiride"],
    "warfarin": ["warfarin"],
    "ciprofloxacin": ["ciprofloxacin"],
    "azithromycin": ["azithromycin"],
    "clopidogrel": ["clopidogrel"],
}


def get_all_aliases(salt: str) -> set[str]:
    """Returns lowercased set of base salt name, original salt name, and all known aliases."""
    base = get_base_salt(salt).strip().lower()
    aliases = {base, salt.strip().lower()}
    if base in DRUG_ALIASES:
        for a in DRUG_ALIASES[base]:
            aliases.add(a.lower())
            aliases.add(get_base_salt(a).lower())
    return aliases


def is_valid_pair_match(query_a: str, query_b: str, row_a: str, row_b: str) -> bool:
    """
    Strict Semantic False-Positive Guard:
    Verifies that returned row drugs actually correspond to the queried salts
    (or their INN/USAN aliases). Discards unrelated vector neighbors.
    """
    aliases_a = get_all_aliases(query_a)
    aliases_b = get_all_aliases(query_b)
    row_a_clean = get_base_salt(row_a).strip().lower()
    row_b_clean = get_base_salt(row_b).strip().lower()

    match_direct = (row_a_clean in aliases_a and row_b_clean in aliases_b)
    match_reverse = (row_a_clean in aliases_b and row_b_clean in aliases_a)
    return match_direct or match_reverse


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLINICAL DRUG CLASSES & ALLERGY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

ALLERGY_DRUG_CLASSES: dict[str, list[str]] = {
    "nsaid": [
        "aspirin", "ibuprofen", "diclofenac", "naproxen", "mefenamic acid",
        "indomethacin", "celecoxib", "etoricoxib", "ketorolac", "piroxicam",
        "ketoprofen", "aceclofenac"
    ],
    "penicillin": [
        "amoxicillin", "amoxycillin", "ampicillin", "penicillin",
        "cloxacillin", "piperacillin", "flucloxacillin"
    ],
    "sulfa": [
        "sulfamethoxazole", "sulfasalazine", "dapsone", "hydrochlorothiazide",
        "furosemide", "glimepiride"
    ],
}


def check_allergies(
    unique_salts: list[str],
    salt_sources: dict[str, list[str]],
    known_allergies: list[str],
) -> list[dict[str, Any]]:
    """
    Cross-checks unique generic salts against known allergies.
    Alias-aware and clinical class-aware (NSAID, Penicillin, Sulfa).
    Returns structured dicts matching Phase 4 synthesizer contract.
    """
    conflicts: list[dict[str, Any]] = []
    if not known_allergies or not unique_salts:
        return conflicts

    for allergen_raw in known_allergies:
        allergen = allergen_raw.strip()
        if not allergen:
            continue
        allergen_lower = allergen.lower()
        allergen_aliases = get_all_aliases(allergen)

        # Normalize class name key (e.g. 'NSAIDs' -> 'nsaid')
        class_key = allergen_lower.rstrip("s")
        class_salts = ALLERGY_DRUG_CLASSES.get(class_key, [])

        for salt in unique_salts:
            salt_base = get_base_salt(salt).strip()
            salt_base_lower = salt_base.lower()
            sources = salt_sources.get(salt, salt_sources.get(salt_base, []))
            source_str = f" (found in {', '.join(sources)})" if sources else ""

            # Check direct or alias match
            if salt_base_lower in allergen_aliases or salt.lower() in allergen_aliases:
                conflicts.append({
                    "allergen": allergen,
                    "matched_salt": salt,
                    "source_brands": sources,
                    "message": f"Patient has a known allergy to {allergen}, which is present as {salt}{source_str}."
                })
            # Check drug class match
            elif salt_base_lower in class_salts:
                conflicts.append({
                    "allergen": allergen,
                    "matched_salt": salt,
                    "source_brands": sources,
                    "message": f"Patient is allergic to {allergen}; {salt}{source_str} belongs to the {allergen} drug class."
                })

    return conflicts


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN INTERACTION CHECKER PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

SEVERITY_ORDER = {"Critical": 0, "Major": 1, "Moderate": 2}


def check_drug_interactions(
    unique_salts: list[str],
    salt_sources: dict[str, list[str]],
    known_allergies: list[str] | None = None,
    embedder: SentenceTransformer | None = None,
    supabase: Client | None = None,
) -> dict[str, Any]:
    """
    Full DDI & allergy safety checker:
      1. Strips counter-ions to base salts
      2. Performs allergy cross-check
      3. Decision Gate 2: len(unique_salts) < 2 -> safe bypass
      4. Exact Mathematical FDC Guard Rule
      5. Batched vectorization of candidate pairs
      6. Supabase hybrid RPC search + Strict Semantic False-Positive Guard
      7. Returns sorted interactions (Critical > Major > Moderate), fdc_notes, allergy_conflicts
    """
    if known_allergies is None:
        known_allergies = []

    # 1. Map each salt to its clean base salt
    base_salt_sources: dict[str, list[str]] = {}
    base_to_original: dict[str, str] = {}

    for salt in unique_salts:
        base = get_base_salt(salt)
        if base not in base_salt_sources:
            base_salt_sources[base] = []
            base_to_original[base] = salt
        # Merge source brands
        sources = salt_sources.get(salt, [])
        for s in sources:
            if s not in base_salt_sources[base]:
                base_salt_sources[base].append(s)

    unique_base_salts = list(base_salt_sources.keys())

    # 2. Check patient allergies
    allergy_conflicts = check_allergies(unique_salts, salt_sources, known_allergies)

    # 3. Decision Gate 2: If fewer than 2 distinct salts, bypass pair checks
    if len(unique_base_salts) < 2:
        return {
            "interactions": [],
            "allergy_conflicts": allergy_conflicts,
            "fdc_notes": [],
            "pairs_checked_count": 0,
            "unique_salts": unique_salts,
            "status": "SAFE" if not allergy_conflicts else "ALLERGY_ALERT",
        }

    # 4. Generate unique combinatorial pairs: nC2
    pairs = list(itertools.combinations(sorted(unique_base_salts), 2))

    # 5. Exact Mathematical FDC Guard Rule
    candidate_pairs: list[tuple[str, str]] = []
    fdc_notes: list[str] = []

    for salt_a, salt_b in pairs:
        set_a = set(base_salt_sources.get(salt_a, []))
        set_b = set(base_salt_sources.get(salt_b, []))

        # Treat as FDC ONLY when len(set_a) == 1 and set_a == set_b
        if len(set_a) == 1 and set_a == set_b:
            brand_name = next(iter(set_a))
            fdc_notes.append(
                f"{salt_a} and {salt_b} are co-formulated in a single tablet ({brand_name}); cross-drug alert skipped."
            )
        else:
            candidate_pairs.append((salt_a, salt_b))

    if not candidate_pairs:
        return {
            "interactions": [],
            "allergy_conflicts": allergy_conflicts,
            "fdc_notes": fdc_notes,
            "pairs_checked_count": 0,
            "unique_salts": unique_salts,
            "status": "SAFE" if not allergy_conflicts else "ALLERGY_ALERT",
        }

    # 6. Batched Vectorization (all candidate queries embedded together)
    queries = [f"Interaction between {a} and {b}" for a, b in candidate_pairs]

    if embedder is None:
        try:
            embedder = SentenceTransformer(settings.embedding_model, local_files_only=True)
        except Exception:
            embedder = SentenceTransformer(settings.embedding_model)

    embeddings = embedder.encode(
        queries,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # 7. Hybrid Supabase Retrieval & Semantic False-Positive Guard
    raw_interactions: list[dict[str, Any]] = []
    seen_row_ids: set[int] = set()

    for i, (salt_a, salt_b) in enumerate(candidate_pairs):
        query_vec = embeddings[i].tolist()

        if supabase is None:
            continue

        rpc_res = supabase.rpc(
            "match_drug_interactions",
            {
                "query_drug_a": salt_a,
                "query_drug_b": salt_b,
                "query_embedding": query_vec,
                "similarity_threshold": settings.similarity_threshold,
                "match_count": settings.rag_match_count,
            }
        ).execute()

        rows = rpc_res.data if rpc_res.data else []

        for row in rows:
            row_id = row.get("id")
            if row_id in seen_row_ids:
                continue

            match_type = row.get("match_type", "exact")

            # Strict Semantic False-Positive Guard:
            # Semantic matches MUST verify that returned row drugs correspond to queried salts or aliases
            if match_type == "semantic":
                if not is_valid_pair_match(salt_a, salt_b, row.get("drug_a", ""), row.get("drug_b", "")):
                    continue  # Discard unrelated vector neighbor!

            seen_row_ids.add(row_id)
            raw_interactions.append({
                "id": row_id,
                "drug_a": row.get("drug_a"),
                "drug_b": row.get("drug_b"),
                "severity": row.get("severity"),
                "clinical_description": row.get("clinical_description"),
                "explanation": row.get("clinical_description"),
                "semantic_chunk": row.get("semantic_chunk"),
                "similarity_score": row.get("similarity_score"),
                "match_type": match_type,
                "query_pair": f"{salt_a} + {salt_b}",
            })

    # 8. Sort interactions: Critical > Major > Moderate, then by similarity score descending
    raw_interactions.sort(
        key=lambda x: (
            SEVERITY_ORDER.get(x["severity"], 3),
            -(x.get("similarity_score") or 0.0)
        )
    )

    # ── True Alias Fallback on RPC Lookup ──
    # If no valid rows found for a candidate pair, retry with INN/USAN alias pair
    if supabase is not None:
        alias_interactions: list[dict[str, Any]] = []
        for i, (salt_a, salt_b) in enumerate(candidate_pairs):
            # Only retry if we got no valid rows for this pair on first pass
            pair_key = f"{salt_a} + {salt_b}"
            has_valid = any(r["query_pair"] == pair_key for r in raw_interactions)
            if has_valid:
                continue

            # Check if either salt has an INN/USAN alias
            aliases_a = get_all_aliases(salt_a)
            aliases_b = get_all_aliases(salt_b)
            alias_a = None
            alias_b = None
            for a in aliases_a:
                if a.lower() != salt_a.lower() and a.lower() != get_base_salt(salt_a).lower():
                    alias_a = a
                    break
            for b in aliases_b:
                if b.lower() != salt_b.lower() and b.lower() != get_base_salt(salt_b).lower():
                    alias_b = b
                    break

            if not alias_a and not alias_b:
                continue

            retry_a = alias_a if alias_a else salt_a
            retry_b = alias_b if alias_b else salt_b

            query_vec = embeddings[i].tolist()
            rpc_res = supabase.rpc(
                "match_drug_interactions",
                {
                    "query_drug_a": retry_a,
                    "query_drug_b": retry_b,
                    "query_embedding": query_vec,
                    "similarity_threshold": settings.similarity_threshold,
                    "match_count": settings.rag_match_count,
                }
            ).execute()

            rows = rpc_res.data if rpc_res.data else []
            for row in rows:
                row_id = row.get("id")
                if row_id in seen_row_ids:
                    continue
                match_type = row.get("match_type", "exact")
                if match_type == "semantic":
                    if not is_valid_pair_match(salt_a, salt_b, row.get("drug_a", ""), row.get("drug_b", "")):
                        # Also check alias pair validity
                        if not is_valid_pair_match(retry_a, retry_b, row.get("drug_a", ""), row.get("drug_b", "")):
                            continue
                seen_row_ids.add(row_id)
                alias_interactions.append({
                    "id": row_id,
                    "drug_a": row.get("drug_a"),
                    "drug_b": row.get("drug_b"),
                    "severity": row.get("severity"),
                    "clinical_description": row.get("clinical_description"),
                    "explanation": row.get("clinical_description"),
                    "semantic_chunk": row.get("semantic_chunk"),
                    "similarity_score": row.get("similarity_score"),
                    "match_type": match_type,
                    "query_pair": pair_key,
                })

        raw_interactions.extend(alias_interactions)

        # Re-sort after adding alias fallback results
        if alias_interactions:
            raw_interactions.sort(
                key=lambda x: (
                    SEVERITY_ORDER.get(x["severity"], 3),
                    -(x.get("similarity_score") or 0.0)
                )
            )

    # ── Status Priority: Allergy conflicts elevated to top alongside CRITICAL ──
    has_critical = any(i["severity"] == "Critical" for i in raw_interactions)
    has_allergy = bool(allergy_conflicts)
    has_major = any(i["severity"] == "Major" for i in raw_interactions)
    has_moderate = any(i["severity"] == "Moderate" for i in raw_interactions)

    if has_critical or has_allergy:
        status = "CRITICAL"
    elif has_major:
        status = "MAJOR"
    elif has_moderate:
        status = "MODERATE"
    else:
        status = "SAFE"

    return {
        "interactions": raw_interactions,
        "allergy_conflicts": allergy_conflicts,
        "fdc_notes": fdc_notes,
        "pairs_checked_count": len(candidate_pairs),
        "unique_salts": unique_salts,
        "status": status,
    }
