"""
FastAPI application entry point.

Key architecture decisions:
  1. Lifespan context manager loads SentenceTransformer ONCE at startup
     (not per-request) → saves ~10-15 seconds per request.
  2. EasyOCR is NOT preloaded here — lazy-loaded in Phase 1 vision_ocr.py
     to save ~1 GB RAM during Phase 4 development.
  3. Models stored in app.state → accessible from any endpoint via request.app.state.
  4. Supabase client also initialized once at startup.
  5. Ollama is NOT loaded here — it runs as a separate process and we call its
     REST API on-demand. We just check if it's reachable at startup.
  6. torch.set_num_threads(4) caps PyTorch CPU threads so embeddings don't freeze.
"""

import httpx
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.supabase_client import get_supabase_client
from app.models.responses import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at server startup (before first request) and once at shutdown.

    Loads SentenceTransformer into memory so it persists across all requests.
    EasyOCR is lazy-loaded in vision_ocr.py on first /upload request.
    """
    print("Starting up — loading models...")

    # ── Cap PyTorch CPU threads to prevent machine freeze ──
    try:
        import torch
        torch.set_num_threads(4)
        print(f"✅ PyTorch CPU threads capped at {torch.get_num_threads()}")
    except ImportError:
        print("⚠️ PyTorch not installed — thread cap skipped")

    # ── EasyOCR: NOT preloaded (lazy-load in Phase 1 to save ~1 GB RAM) ──
    app.state.ocr_reader = None  # Will be lazy-loaded by vision_ocr.py
    print("ℹ️ EasyOCR deferred (lazy-load on first /upload)")

    # ── Load SentenceTransformer for local embeddings ──
    from sentence_transformers import SentenceTransformer
    app.state.embedder = SentenceTransformer(settings.embedding_model)
    print(f"✅ SentenceTransformer ({settings.embedding_model}) loaded")

    # ── Initialize Supabase client ──
    app.state.supabase = get_supabase_client()
    print("✅ Supabase client initialized")

    # ── Check if Ollama is reachable ──
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            app.state.ollama_available = True
            print(f"✅ Ollama is running ({settings.ollama_base_url})")
    except Exception:
        app.state.ollama_available = False
        print(f"⚠️ Ollama not reachable at {settings.ollama_base_url} — will use template fallback")

    print("\nAll models loaded. Server is ready.\n")

    yield  # Server runs and handles requests here

    # ── Shutdown cleanup ──
    print("Shutting down...")
    # Models are garbage collected automatically


# ── Create the FastAPI app ──

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Privacy-first polypharmacy drug-drug interaction checker for Indian prescriptions",
    lifespan=lifespan,
)

# ── CORS middleware (allow Next.js frontend to call us) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Next.js dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ──
from app.routers.prescription import router as prescription_router  # noqa: E402
app.include_router(prescription_router)


# ── Health check endpoint ──

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Quick health check. Verifies:
      - Server is running
      - ML models are loaded in memory (embedder required, OCR optional)
      - Ollama is reachable (optional)
    """
    embedder_loaded = (
        hasattr(app.state, "embedder")
        and app.state.embedder is not None
    )

    # Re-check Ollama availability (it might have started/stopped since boot)
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok" if embedder_loaded else "degraded",
        version=settings.app_version,
        models_loaded=embedder_loaded,
        ollama_available=ollama_ok,
    )

