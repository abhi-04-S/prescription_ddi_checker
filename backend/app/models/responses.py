"""
Pydantic v2 response models — define the shape of every API response.

These schemas are used for:
  1. FastAPI auto-generates OpenAPI docs from these
  2. The synthesizer validates LLM output against these (if Ollama returns bad JSON, we catch it)
  3. The frontend TypeScript types can be auto-generated from the OpenAPI spec
"""

from pydantic import BaseModel, Field
from enum import Enum


# ── Enums ──

class SeverityLevel(str, Enum):
    """Interaction severity, ordered from most to least dangerous."""
    CRITICAL = "Critical"
    MAJOR = "Major"
    MODERATE = "Moderate"
    SAFE = "Safe"


class ScanStatus(str, Enum):
    """Status of the OCR scan step."""
    SUCCESS = "SUCCESS"                       # All lines extracted with high confidence
    VERIFICATION_REQUIRED = "VERIFICATION_REQUIRED"  # Some lines need human review
    FAILED = "FAILED"                         # OCR completely failed


# ── OCR / Upload Response Models ──

class ExtractedMedicine(BaseModel):
    """A single medicine line extracted by OCR."""
    raw_text: str = Field(..., description="Original text as read by OCR")
    confidence: float = Field(..., description="Mean OCR confidence for this line (0-100)")
    needs_review: bool = Field(False, description="True if confidence < threshold")


class PrescriptionUploadResponse(BaseModel):
    """Response from POST /api/v1/prescription/upload"""
    status: ScanStatus
    medicines: list[ExtractedMedicine] = []
    message: str = ""


# ── Normalization Models ──

class ResolvedMedicine(BaseModel):
    """A brand name resolved to its generic salts."""
    original_text: str = Field(..., description="The cleaned brand name from OCR")
    brand_name: str | None = Field(None, description="Matched brand name from DB/CSV")
    active_salts: list[str] = Field(default_factory=list, description="Generic salt names")
    strength: str | None = None
    match_method: str | None = Field(None, description="'csv_exact', 'db_exact', 'db_fuzzy', or 'unresolved'")
    match_score: float | None = Field(None, description="Fuzzy match score if applicable")


# ── Interaction / RAG Response Models ──

class InteractionDetail(BaseModel):
    """A single drug-drug interaction found by the RAG checker."""
    drug_a: str
    drug_b: str
    severity: str
    explanation: str = Field(..., description="Plain-language explanation of the interaction")
    match_type: str | None = Field(None, description="'exact' or 'semantic' from hybrid search")
    similarity_score: float | None = None


class SafetyReportResponse(BaseModel):
    """
    The final safety report returned to the user.
    This is what the frontend renders as the report card.
    """
    overall_severity: str = Field(
        ...,
        description="Worst severity found: Critical > Major > Moderate > Safe"
    )
    interactions: list[InteractionDetail] = Field(
        default_factory=list,
        description="List of found drug-drug interactions"
    )
    duplicate_salt_warnings: list[str] = Field(
        default_factory=list,
        description="Salts that appear in multiple brands (overdose risk)"
    )
    allergy_warnings: list[str] = Field(
        default_factory=list,
        description="Salts that conflict with user's known allergies"
    )
    fdc_notes: list[str] = Field(
        default_factory=list,
        description="Fixed-Dose Combination notes (co-formulated salts in one tablet)"
    )
    resolved_medicines: list[ResolvedMedicine] = Field(
        default_factory=list,
        description="All medicines resolved to their generic salts"
    )
    unresolved_medicines: list[str] = Field(
        default_factory=list,
        description="Medicine lines that could not be resolved"
    )
    synthesis_mode: str = Field(
        default="deterministic_safe",
        description="'deterministic_safe', 'ollama_phi35', or 'template_fallback'"
    )
    disclaimer: str = "This is an informational tool, not medical advice. Always consult your doctor."


# ── Analyze Request Model ──

class AnalyzeRequest(BaseModel):
    """Request body for POST /api/v1/prescription/analyze"""
    medicines: list[str] = Field(
        ...,
        min_length=1,
        description="List of confirmed medicine strings (e.g., ['Dolo 650', 'Brufen 400'])"
    )
    known_allergies: list[str] = Field(
        default_factory=list,
        description="Optional list of known drug allergies (generic salt names)"
    )


# ── Health Check ──

class HealthResponse(BaseModel):
    """Response from GET /health"""
    status: str = "ok"
    version: str
    models_loaded: bool
    ollama_available: bool

