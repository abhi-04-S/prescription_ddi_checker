"""
Supabase client singleton — one connection shared across the app.

Uses the official supabase-py async client. Initialized once at startup
via FastAPI's lifespan, stored in app.state, accessed by services.
"""

from supabase import create_client, Client
from app.config import settings


def get_supabase_client() -> Client:
    """
    Create and return a Supabase client instance.
    Called once during app startup and stored in app.state.supabase.
    """
    return create_client(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_key,
    )

