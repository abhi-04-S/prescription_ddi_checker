"""
Prescription Router — POST /api/v1/prescription/analyze

Full pipeline: normalize → check → synthesize
Chains salt_normalizer → rag_checker → synthesizer services.
"""

from fastapi import APIRouter, Request

from app.models.requests import AnalyzePrescriptionRequest
from app.models.responses import SafetyReportResponse
from app.services.salt_normalizer import normalize_prescription
from app.services.rag_checker import check_drug_interactions
from app.services.synthesizer import synthesize_report
from app.config import settings

router = APIRouter(prefix="/api/v1/prescription", tags=["Prescription"])


@router.post("/analyze", response_model=SafetyReportResponse)
async def analyze_prescription(
    body: AnalyzePrescriptionRequest,
    request: Request,
):
    """
    Accepts a list of prescription medicine strings + optional known allergies.
    Returns a full safety report with interactions, severity, and explanations.

    Pipeline:
      1. salt_normalizer.normalize_prescription() → resolves brands to salts
      2. rag_checker.check_drug_interactions() → finds DDI pairs + allergies
      3. synthesizer.synthesize_report() → 3-tier LLM/template report
    """
    # ── Step 1: Normalize all medicine lines ──
    supabase = request.app.state.supabase
    norm_result = normalize_prescription(
        medicine_lines=body.medicine_lines,
        supabase=supabase,
        similarity_cutoff=settings.brand_similarity_cutoff,
    )

    # ── Step 2: Check drug interactions + allergies ──
    embedder = request.app.state.embedder
    rag_result = check_drug_interactions(
        unique_salts=norm_result["unique_salts"],
        salt_sources=norm_result["salt_sources"],
        known_allergies=body.known_allergies,
        embedder=embedder,
        supabase=supabase,
    )

    # ── Step 3: Synthesize report (3-tier: safe bypass / Ollama / template) ──
    report = await synthesize_report(
        interactions=rag_result["interactions"],
        duplicate_salts=norm_result["duplicate_salt_warnings"],
        allergy_conflicts=rag_result["allergy_conflicts"],
        fdc_notes=rag_result["fdc_notes"],
        resolved_medicines=norm_result["resolved_medicines"],
        unresolved_medicines=norm_result["unresolved_medicines"],
    )

    return report
