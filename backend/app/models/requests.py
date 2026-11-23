"""
Pydantic v2 request models — define the shape of every API request body.
"""

from pydantic import BaseModel, Field


class AnalyzePrescriptionRequest(BaseModel):
    """Request body for POST /api/v1/prescription/analyze"""
    medicine_lines: list[str] = Field(
        ...,
        min_length=1,
        description="List of raw prescription strings (e.g., ['Tab Dolo 650 OD', 'Cap Telma 40 BD'])"
    )
    known_allergies: list[str] = Field(
        default_factory=list,
        description="Optional list of known drug allergies (generic salt or class names)"
    )
