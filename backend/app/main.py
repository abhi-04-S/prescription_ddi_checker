"""
FastAPI application entry point.

Key architecture decisions:
  1. Lifespan context manager loads EasyOCR + SentenceTransformer ONCE at startup
     (not per-request) → saves ~10-15 seconds per request.
  2. Models stored in app.state → accessible from any endpoint via request.app.state.
  3. Supabase client also initialized once at startup.
  4. Ollama is NOT loaded here — it runs as a separate process and we call its
     REST API on-demand. We just check if it's reachable at startup.
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

    Loads heavy ML models into memory so they persist across all requests.
    Without this, each request would re-load ~400MB of models = 10-15s delay.
    """
    print("🚀 Starting up — loading models...")

    # ── Load EasyOCR reader (English) ──
    # Lazy import: only loads the library when the server starts
    import easyocr
    app.state.ocr_reader = easyocr.Reader(
        ["en"],
        gpu=settings.ocr_gpu,
        verbose=False,
    )
    print("✅ EasyOCR loaded")

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
        print(f"⚠️  Ollama not reachable at {settings.ollama_base_url} — will use template fallback")

    print("🟢 All models loaded. Server is ready.\n")

    yield  # ← Server runs and handles requests here

    # ── Shutdown cleanup ──
    print("🔴 Shutting down...")
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


# ── Health check endpoint ──

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Quick health check. Verifies:
      - Server is running
      - ML models are loaded in memory
      - Ollama is reachable (optional)
    """
    models_loaded = (
        hasattr(app.state, "ocr_reader")
        and hasattr(app.state, "embedder")
        and app.state.ocr_reader is not None
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
        status="ok" if models_loaded else "degraded",
        version=settings.app_version,
        models_loaded=models_loaded,
        ollama_available=ollama_ok,
    )

