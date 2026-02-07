"""LangGraph shared state for the Eunomia V2 orchestrator.

Uses TypedDict so each key becomes its own LangGraph channel with
proper partial-update merge semantics. Nodes return partial dicts
and LangGraph replaces only the keys present in the return value.
"""

from enum import Enum
from typing import Any, TypedDict


DEFAULT_MODEL = "anthropic:claude-sonnet-4-5-20250929"


class HITLLevel(str, Enum):
    """Human-in-the-loop oversight level."""

    AUTONOMOUS = "autonomous"    # No interrupts (default, backward compatible)
    APPROVAL = "approval"        # Interrupt after plan + on escalation
    INTERACTIVE = "interactive"  # Interrupt before every task + commit


class EunomiaState(TypedDict, total=False):
    """State passed between all nodes in the LangGraph graph.

    Each key is a separate LangGraph channel (replace reducer).
    Nodes return partial dicts — only changed keys.
    """

    project_path: str
    prd_content: str
    tasks: list                  # list[Task] — Pydantic Task objects
    current_task_index: int
    developer_output: Any        # DeveloperOutput | None
    qa_result: Any               # QAResult | None
    architect_decisions: list    # list[ArchitectDecision]
    retry_count: int
    max_retries: int
    completed_tasks: list        # list[str] — task IDs as strings
    failed_tasks: list           # list[str] — task IDs as strings
    session_id: str
    model: str
    # HITL fields (M8)
    hitl_level: str              # HITLLevel value string
    cached_plan: list            # Cached task list (avoids re-calling LLM on resume)
    user_feedback: str           # Last user feedback from interrupt


def create_initial_state(
    project_path: str,
    prd_content: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    session_id: str = "",
    hitl_level: str = "autonomous",
) -> dict[str, Any]:
    """Create the initial state for a new pipeline run."""
    return {
        "project_path": project_path,
        "prd_content": prd_content,
        "tasks": [],
        "current_task_index": 0,
        "developer_output": None,
        "qa_result": None,
        "architect_decisions": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "completed_tasks": [],
        "failed_tasks": [],
        "session_id": session_id,
        "model": model,
        "hitl_level": hitl_level,
        "cached_plan": [],
        "user_feedback": "",
    }
