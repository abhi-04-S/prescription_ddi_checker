"""
Application configuration — loads all settings from environment variables.

Uses pydantic-settings so you define config once here and it auto-reads from
.env file or system environment variables. No hardcoded secrets anywhere.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    All configuration lives here. Values come from environment variables
    or the .env file in the backend/ directory.
    """

    # ── App ──
    app_name: str = "Prescription DDI Checker"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── Supabase ──
    # You'll get these from your Supabase project dashboard:
    # Settings → API → Project URL and anon/service_role key
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_key: str = Field(..., description="Supabase anon or service_role key")

    # ── Ollama (Local LLM) ──
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3.5"
    ollama_timeout_seconds: float = 120.0

    # ── OCR ──
    ocr_confidence_threshold: float = 75.0  # Below this → ask user to verify
    ocr_gpu: bool = False                   # Set True if you have a CUDA GPU

    # ── RAG ──
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.82      # Minimum cosine similarity for vector matches
    rag_match_count: int = 5                # Max results per pair from hybrid search

    # ── Brand Lookup ──
    # Fuzzy matching threshold for pg_trgm (0.0 to 1.0, higher = stricter)
    brand_similarity_cutoff: float = 0.3

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton instance — import this everywhere
settings = Settings()

