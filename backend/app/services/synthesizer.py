"""
Synthesizer Service — 3-Tier Local Ollama Synthesis + Deterministic Safety Overrides.

Tier 1: Deterministic SAFE bypass (zero LLM usage)
Tier 2: Local Ollama Phi-3.5 via httpx (resource-safe options)
Tier 3: Graceful template fallback (Ollama unreachable)

After validating phi3.5's JSON response with Pydantic, ALWAYS overwrites
overall_severity, duplicate_salt_warnings, allergy_warnings, fdc_notes
deterministically in Python so the 3.8B model can never downgrade a risk.
"""

import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.models.responses import (
    SafetyReportResponse,
    InteractionDetail,
    ResolvedMedicine,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SEVERITY RANKING & DETERMINISTIC OVERRIDE
# ═══════════════════════════════════════════════════════════════════════════════

SEVERITY_RANK = {"Critical": 3, "Major": 2, "Moderate": 1, "Safe": 0}


def compute_overall_severity(
    interactions: list[dict[str, Any]],
    duplicate_salts: list[str],
    allergy_conflicts: list[dict[str, Any]],
) -> str:
    """
    Deterministic severity computation — Python ALWAYS overrides LLM:
      - Any Critical DDI OR any allergy_conflicts → "Critical"
      - Else any Major DDI OR any duplicate_salts → "Major"
      - Else any Moderate DDI → "Moderate"
      - Else → "Safe"
    """
    has_critical = any(i.get("severity") == "Critical" for i in interactions)
    has_allergy = bool(allergy_conflicts)
    has_major = any(i.get("severity") == "Major" for i in interactions)
    has_duplicate = bool(duplicate_salts)
    has_moderate = any(i.get("severity") == "Moderate" for i in interactions)

    if has_critical or has_allergy:
        return "Critical"
    if has_major or has_duplicate:
        return "Major"
    if has_moderate:
        return "Moderate"
    return "Safe"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LAYPERSON TEMPLATES (Tier 3 — Template Fallback)
# ═══════════════════════════════════════════════════════════════════════════════

LAYPERSON_TEMPLATES = {
    "Critical": "⛔ {drug_a} and {drug_b} can be dangerous together. {desc} Talk to your doctor before taking both.",
    "Major":    "⚠️ {drug_a} and {drug_b} may cause problems together. {desc} Ask your pharmacist.",
    "Moderate": "ℹ️ {drug_a} and {drug_b} have a mild interaction. {desc} Usually safe under doctor supervision.",
}


def _build_template_explanations(interactions: list[dict[str, Any]]) -> list[InteractionDetail]:
    """Build InteractionDetail list from raw interactions using templates."""
    details: list[InteractionDetail] = []
    for row in interactions:
        severity = row.get("severity", "Moderate")
        tmpl = LAYPERSON_TEMPLATES.get(severity, LAYPERSON_TEMPLATES["Moderate"])
        details.append(InteractionDetail(
            drug_a=row["drug_a"],
            drug_b=row["drug_b"],
            severity=severity,
            explanation=tmpl.format(
                drug_a=row["drug_a"],
                drug_b=row["drug_b"],
                desc=row.get("clinical_description", row.get("explanation", "")),
            ),
            match_type=row.get("match_type"),
            similarity_score=row.get("similarity_score"),
        ))
    # Sort: Critical > Major > Moderate
    details.sort(key=lambda d: -SEVERITY_RANK.get(d.severity, 0))
    return details


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OLLAMA SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a clinical pharmacist translator.
You will receive ONLY verified drug interaction records from a medical database, plus any duplicate salt warnings and allergy conflicts.
Explain each interaction in simple, non-alarmist language for an elderly patient or caregiver.
STRICTLY use only the provided data. NEVER invent interactions, dosages, or side effects.
Output valid JSON matching this exact structure:
{
  "interactions": [
    {
      "drug_a": "salt name",
      "drug_b": "salt name",
      "severity": "Critical|Major|Moderate",
      "explanation": "plain language explanation in 1-2 sentences"
    }
  ],
  "general_advice": "one brief sentence of overall advice"
}"""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAIN SYNTHESIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

async def synthesize_report(
    interactions: list[dict[str, Any]],
    duplicate_salts: list[str],
    allergy_conflicts: list[dict[str, Any]],
    fdc_notes: list[str] | None = None,
    resolved_medicines: list[ResolvedMedicine] | None = None,
    unresolved_medicines: list[str] | None = None,
) -> SafetyReportResponse:
    """
    3-Tier synthesis pipeline:

    Tier 1 — Deterministic SAFE Bypass (Decision Gate 3):
        If interactions, duplicate_salts, and allergy_conflicts are ALL empty,
        immediately return SAFE without calling Ollama (0 MB LLM RAM used).

    Tier 2 — Local Ollama phi3.5 via httpx:
        POST to Ollama REST API with resource-safe options.
        After Pydantic validation, deterministically overwrite severity fields.

    Tier 3 — Graceful Template Fallback:
        On ConnectError / TimeoutException, generate template-based report.
    """
    if fdc_notes is None:
        fdc_notes = []
    if resolved_medicines is None:
        resolved_medicines = []
    if unresolved_medicines is None:
        unresolved_medicines = []

    # ── Deterministic severity (ALWAYS used, never trust LLM for this) ──
    overall_severity = compute_overall_severity(interactions, duplicate_salts, allergy_conflicts)

    # ── Build deterministic allergy warnings ──
    allergy_warnings = [c["message"] for c in allergy_conflicts]

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 1 — Deterministic SAFE Bypass
    # ═══════════════════════════════════════════════════════════════════════
    if not interactions and not duplicate_salts and not allergy_conflicts:
        return SafetyReportResponse(
            overall_severity="Safe",
            interactions=[],
            duplicate_salt_warnings=[],
            allergy_warnings=[],
            fdc_notes=fdc_notes,
            resolved_medicines=resolved_medicines,
            unresolved_medicines=unresolved_medicines,
            synthesis_mode="deterministic_safe",
            disclaimer="No interactions found. This is informational, not medical advice. Always consult your doctor.",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 2 — Local Ollama Phi-3.5
    # ═══════════════════════════════════════════════════════════════════════
    try:
        prompt_data = {
            "verified_interactions": [
                {
                    "drug_a": i["drug_a"],
                    "drug_b": i["drug_b"],
                    "severity": i["severity"],
                    "clinical_description": i.get("clinical_description", i.get("explanation", "")),
                }
                for i in interactions
            ],
            "duplicate_salts": duplicate_salts,
            "allergy_conflicts": [
                {"allergen": c["allergen"], "matched_salt": c["matched_salt"], "message": c["message"]}
                for c in allergy_conflicts
            ],
        }

        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            resp = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "system": SYSTEM_PROMPT,
                    "prompt": json.dumps(prompt_data),
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 2048,
                        "num_predict": 512,
                        "num_thread": 6,
                    },
                },
            )
            resp.raise_for_status()

        llm_json = resp.json().get("response", "{}")

        # Parse LLM response
        try:
            llm_data = json.loads(llm_json)
        except json.JSONDecodeError:
            logger.warning("Ollama returned invalid JSON, falling back to template")
            raise ValueError("Invalid JSON from Ollama")

        # Build interaction details from LLM explanations
        llm_interactions = llm_data.get("interactions", [])
        interaction_details: list[InteractionDetail] = []

        for llm_item in llm_interactions:
            # Find matching raw interaction for metadata (match_type, similarity_score)
            raw_match = next(
                (r for r in interactions
                 if r["drug_a"].lower() == llm_item.get("drug_a", "").lower()
                 and r["drug_b"].lower() == llm_item.get("drug_b", "").lower()),
                None,
            )
            if raw_match is None:
                # Try reverse order
                raw_match = next(
                    (r for r in interactions
                     if r["drug_a"].lower() == llm_item.get("drug_b", "").lower()
                     and r["drug_b"].lower() == llm_item.get("drug_a", "").lower()),
                    None,
                )

            interaction_details.append(InteractionDetail(
                drug_a=llm_item.get("drug_a", "Unknown"),
                drug_b=llm_item.get("drug_b", "Unknown"),
                severity=llm_item.get("severity", "Moderate"),
                explanation=llm_item.get("explanation", ""),
                match_type=raw_match.get("match_type") if raw_match else None,
                similarity_score=raw_match.get("similarity_score") if raw_match else None,
            ))

        # If LLM missed some interactions, add them from template
        llm_pairs = {
            (d.drug_a.lower(), d.drug_b.lower()) for d in interaction_details
        } | {
            (d.drug_b.lower(), d.drug_a.lower()) for d in interaction_details
        }
        for raw in interactions:
            pair = (raw["drug_a"].lower(), raw["drug_b"].lower())
            if pair not in llm_pairs:
                tmpl = LAYPERSON_TEMPLATES.get(raw["severity"], LAYPERSON_TEMPLATES["Moderate"])
                interaction_details.append(InteractionDetail(
                    drug_a=raw["drug_a"],
                    drug_b=raw["drug_b"],
                    severity=raw["severity"],
                    explanation=tmpl.format(
                        drug_a=raw["drug_a"],
                        drug_b=raw["drug_b"],
                        desc=raw.get("clinical_description", ""),
                    ),
                    match_type=raw.get("match_type"),
                    similarity_score=raw.get("similarity_score"),
                ))

        # Sort: Critical > Major > Moderate
        interaction_details.sort(key=lambda d: -SEVERITY_RANK.get(d.severity, 0))

        # ── DETERMINISTIC OVERRIDE — Python ALWAYS has final say ──
        return SafetyReportResponse(
            overall_severity=overall_severity,         # Python-computed, never LLM
            interactions=interaction_details,
            duplicate_salt_warnings=duplicate_salts,   # Python-computed, never LLM
            allergy_warnings=allergy_warnings,         # Python-computed, never LLM
            fdc_notes=fdc_notes,                       # Python-computed, never LLM
            resolved_medicines=resolved_medicines,
            unresolved_medicines=unresolved_medicines,
            synthesis_mode="ollama_phi35",
            disclaimer="This is an informational tool, not medical advice. Always consult your doctor.",
        )

    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, ValueError) as e:
        logger.warning(f"Ollama unavailable ({e}), using template fallback")

    except Exception as e:
        logger.error(f"Unexpected error calling Ollama: {e}", exc_info=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 3 — Graceful Template Fallback
    # ═══════════════════════════════════════════════════════════════════════
    interaction_details = _build_template_explanations(interactions)

    return SafetyReportResponse(
        overall_severity=overall_severity,
        interactions=interaction_details,
        duplicate_salt_warnings=duplicate_salts,
        allergy_warnings=allergy_warnings,
        fdc_notes=fdc_notes,
        resolved_medicines=resolved_medicines,
        unresolved_medicines=unresolved_medicines,
        synthesis_mode="template_fallback",
        disclaimer="This is an informational tool, not medical advice. Always consult your doctor.",
    )
