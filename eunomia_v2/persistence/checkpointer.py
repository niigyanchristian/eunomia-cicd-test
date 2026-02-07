"""Checkpointer factory — creates MemorySaver or AsyncSqliteSaver based on config."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".eunomia" / "eunomia.db"


def get_checkpointer(
    backend: str = "memory",
    db_path: Path | None = None,
) -> Any:
    """Create a LangGraph checkpointer.

    Must be called from within an async context (running event loop) when
    backend="sqlite", because AsyncSqliteSaver requires one.

    Args:
        backend: "memory" (default, in-process only) or "sqlite" (persistent).
        db_path: SQLite database path. Defaults to ~/.eunomia/eunomia.db.

    Returns:
        A LangGraph-compatible checkpointer (MemorySaver or AsyncSqliteSaver).
    """
    if backend == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path = db_path or DEFAULT_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Using SQLite checkpointer: %s", path)
        conn = aiosqlite.connect(str(path))
        return AsyncSqliteSaver(conn)

    from langgraph.checkpoint.memory import MemorySaver

    logger.debug("Using in-memory checkpointer")
    return MemorySaver()
