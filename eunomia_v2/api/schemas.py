"""Pydantic request/response models for the Eunomia V2 REST API.

Defines all DTOs for session management, task inspection,
WebSocket message framing, and HITL chat interactions.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SessionStatus(str, Enum):
    """Pipeline session lifecycle status."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"          # Waiting for HITL input
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WSMessageType(str, Enum):
    """WebSocket message types (both directions)."""

    EVENT = "event"            # Server → Client: pipeline event
    INTERRUPT = "interrupt"    # Server → Client: HITL interrupt
    CHAT = "chat"              # Client → Server: HITL response
    ERROR = "error"            # Server → Client: error
    PING = "ping"              # Client → Server: keepalive
    PONG = "pong"              # Server → Client: keepalive ack


# ---------------------------------------------------------------------------
# Session requests
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    """Request body for POST /api/v1/sessions."""

    project_path: str = Field(..., description="Absolute path to the project directory")
    prd_content: str = Field(..., min_length=1, description="PRD markdown content")
    model: str = Field(
        default="anthropic:claude-sonnet-4-5-20250929",
        description="LLM model identifier (provider:model_name)",
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Max QA-Dev retry loops")
    hitl_level: str = Field(
        default="autonomous",
        description="HITL oversight level: autonomous, approval, or interactive",
    )


# ---------------------------------------------------------------------------
# Session responses
# ---------------------------------------------------------------------------

class TaskResponse(BaseModel):
    """Single task in the pipeline."""

    id: int
    title: str
    description: str = ""
    task_type: str = ""
    status: str = "pending"
    assigned_agent: str = ""
    retry_count: int = 0
    output: str = ""
    error: str = ""


class CreateSessionResponse(BaseModel):
    """Response for POST /api/v1/sessions."""

    session_id: str
    status: SessionStatus = SessionStatus.CREATED
    message: str = "Session created"


class SessionStatusResponse(BaseModel):
    """Response for GET /api/v1/sessions/{id} and action endpoints."""

    session_id: str
    status: SessionStatus
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    current_task_index: int = 0
    model: str = ""
    hitl_level: str = "autonomous"
    has_interrupt: bool = False
    interrupt_data: dict[str, Any] | None = None


class SessionSummary(BaseModel):
    """Brief session info for list endpoint."""

    session_id: str
    status: SessionStatus
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    model: str = ""


class SessionListResponse(BaseModel):
    """Response for GET /api/v1/sessions."""

    sessions: list[SessionSummary] = Field(default_factory=list)
    count: int = 0


class TaskListResponse(BaseModel):
    """Response for GET /api/v1/sessions/{id}/tasks."""

    session_id: str
    tasks: list[TaskResponse] = Field(default_factory=list)


class ArtifactResponse(BaseModel):
    """Response for GET /api/v1/sessions/{id}/artifacts."""

    session_id: str
    project_path: str = ""
    files_created: list[str] = Field(default_factory=list)
    architect_decisions: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Chat / HITL
# ---------------------------------------------------------------------------

class ChatMessageRequest(BaseModel):
    """Request body for POST /api/v1/sessions/{id}/messages."""

    message: str = Field(..., min_length=1, description="User response to interrupt or chat input")


class ChatMessageResponse(BaseModel):
    """Response for POST /api/v1/sessions/{id}/messages."""

    session_id: str
    accepted: bool = True
    message: str = "Message delivered"
    status: SessionStatus = SessionStatus.RUNNING


# ---------------------------------------------------------------------------
# WebSocket framing
# ---------------------------------------------------------------------------

class WSMessage(BaseModel):
    """WebSocket message envelope (JSON frame)."""

    type: WSMessageType
    data: dict[str, Any] = Field(default_factory=dict)
    index: int | None = None   # Monotonic event index for replay


# ---------------------------------------------------------------------------
# Document generation
# ---------------------------------------------------------------------------

class GenerateDocRequest(BaseModel):
    """Request body for POST /api/v1/sessions/{id}/documents/generate."""

    doc_type: str = Field(..., description="Document type: prd, tdd, adr, readme, test_plan, runbook")
    context: dict[str, Any] = Field(default_factory=dict, description="Context variables for template interpolation")
    model: str = Field(
        default="anthropic:claude-sonnet-4-5-20250929",
        description="LLM model identifier (provider:model_name)",
    )
    review: bool = Field(default=True, description="Whether to run a review pass on each section")


class GenerateDocResponse(BaseModel):
    """Response for POST /api/v1/sessions/{id}/documents/generate."""

    session_id: str
    doc_type: str
    content: str = ""
    file_path: str = ""
    sections_count: int = 0
