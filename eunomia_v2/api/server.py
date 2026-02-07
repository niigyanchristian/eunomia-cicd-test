"""FastAPI server for Eunomia V2.

Wires session REST endpoints, WebSocket streaming, auth, and lifespan.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from eunomia_v2.api.routes.sessions import router as sessions_router
from eunomia_v2.api.routes.streaming import router as streaming_router
from eunomia_v2.api.session_manager import SessionManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage SessionManager lifecycle and periodic cleanup."""
    # Startup
    manager = SessionManager()
    app.state.session_manager = manager
    logger.info("SessionManager initialized")

    # Periodic cleanup task
    async def _cleanup_loop() -> None:
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            try:
                manager.cleanup_expired()
            except Exception:
                logger.exception("Session cleanup error")

    cleanup_task = asyncio.create_task(_cleanup_loop(), name="session-cleanup")

    yield

    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    await manager.shutdown()
    logger.info("SessionManager shut down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Eunomia V2",
    description="Autonomous multi-agent software development framework API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure per environment via EUNOMIA_V2_CORS_ORIGINS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sessions_router)
app.include_router(streaming_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/v1/models")
async def list_models() -> dict[str, list[str]]:
    """List available LLM providers."""
    return {
        "models": [
            "anthropic/claude-sonnet-4-5-20250929",
            "anthropic/claude-opus-4-6",
            "openai/gpt-5.2",
            "google/gemini-3-pro",
        ]
    }
