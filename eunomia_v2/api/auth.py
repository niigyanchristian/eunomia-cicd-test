"""API key authentication and rate limiting for Eunomia V2 API.

- API key auth: reads EUNOMIA_API_KEY env var. If unset, auth is disabled.
- Rate limiter: per-session sliding window (configurable, default 60 req/min).
- WebSocket auth: via ?api_key= query parameter.
"""

import os
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Query, Request, WebSocket, status
from fastapi.security import APIKeyHeader

# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_EXPECTED_KEY: str | None = os.environ.get("EUNOMIA_API_KEY")


def _get_expected_key() -> str | None:
    """Return the expected API key (re-reads env on each call for testability)."""
    return os.environ.get("EUNOMIA_API_KEY")


async def verify_api_key(api_key: str | None = Depends(_api_key_header)) -> str | None:
    """FastAPI dependency — validates X-API-Key header.

    If EUNOMIA_API_KEY is not set, auth is disabled (returns None).
    If set, the header must match or a 401 is raised.
    """
    expected = _get_expected_key()
    if expected is None:
        return None  # Auth disabled
    if not api_key or api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key


async def verify_ws_api_key(
    websocket: WebSocket,
    api_key: str | None = Query(default=None),
) -> str | None:
    """WebSocket auth via ?api_key= query parameter.

    If EUNOMIA_API_KEY is not set, auth is disabled.
    If set and missing/invalid, closes the WebSocket with 4401.
    """
    expected = _get_expected_key()
    if expected is None:
        return None
    if not api_key or api_key != expected:
        await websocket.close(code=4401, reason="Invalid or missing API key")
        raise HTTPException(status_code=401, detail="Unauthorized WebSocket")
    return api_key


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Per-session sliding-window rate limiter.

    Tracks request timestamps per session_id. Rejects requests
    that exceed max_requests within the window_seconds.
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def check(self, session_id: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        now = time.monotonic()
        window = self._windows[session_id]

        # Evict expired timestamps
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self.max_requests:
            return False

        window.append(now)
        return True

    def reset(self, session_id: str) -> None:
        """Clear rate limit state for a session."""
        self._windows.pop(session_id, None)


# Global rate limiter instance
rate_limiter = RateLimiter()


async def check_rate_limit(request: Request) -> None:
    """FastAPI dependency — enforces rate limiting on session endpoints.

    Extracts session_id from path parameters. If absent, uses client IP.
    """
    session_id = request.path_params.get("session_id", request.client.host if request.client else "unknown")
    if not rate_limiter.check(session_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )
