"""REST session endpoints for Eunomia V2 API.

All endpoints are prefixed with /api/v1 via the router.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status

from eunomia_v2.api.auth import check_rate_limit, verify_api_key
from eunomia_v2.api.schemas import (
    ArtifactResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    GenerateDocRequest,
    GenerateDocResponse,
    SessionListResponse,
    SessionStatus,
    SessionStatusResponse,
    SessionSummary,
    TaskListResponse,
    TaskResponse,
)
from eunomia_v2.api.session_manager import SessionManager

router = APIRouter(prefix="/api/v1", tags=["sessions"])


def _get_manager(request: Request) -> SessionManager:
    """Extract SessionManager from app state."""
    return request.app.state.session_manager


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_api_key)],
)
async def create_session(
    body: CreateSessionRequest,
    request: Request,
) -> CreateSessionResponse:
    """Create a new pipeline session."""
    manager = _get_manager(request)
    session = manager.create_session(
        project_path=body.project_path,
        prd_content=body.prd_content,
        model=body.model,
        max_retries=body.max_retries,
        hitl_level=body.hitl_level,
    )
    return CreateSessionResponse(
        session_id=session.session_id,
        status=session.status,
    )


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    dependencies=[Depends(verify_api_key)],
)
async def list_sessions(request: Request) -> SessionListResponse:
    """List all sessions (in-memory active + historical from SQLite)."""
    manager = _get_manager(request)
    summaries = []
    seen_ids: set[str] = set()

    # Active in-memory sessions first
    for s in manager.list_sessions():
        state = s.final_state or {}
        summaries.append(SessionSummary(
            session_id=s.session_id,
            status=s.status,
            total_tasks=len(state.get("tasks", [])),
            completed_tasks=len(state.get("completed_tasks", [])),
            failed_tasks=len(state.get("failed_tasks", [])),
            model=s.model,
        ))
        seen_ids.add(s.session_id)

    # Historical sessions from SQLite
    for row in manager.list_all_sessions(limit=50):
        sid = row.get("session_id", "")
        if sid in seen_ids:
            continue
        completed = row.get("completed_tasks", [])
        failed = row.get("failed_tasks", [])
        summaries.append(SessionSummary(
            session_id=sid,
            status=row.get("status", "unknown"),
            total_tasks=row.get("total_tasks", 0),
            completed_tasks=len(completed) if isinstance(completed, list) else 0,
            failed_tasks=len(failed) if isinstance(failed, list) else 0,
            model=row.get("model", ""),
        ))

    return SessionListResponse(sessions=summaries, count=len(summaries))


@router.get(
    "/sessions/{session_id}",
    response_model=SessionStatusResponse,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def get_session(session_id: str, request: Request) -> SessionStatusResponse:
    """Get detailed session status."""
    manager = _get_manager(request)
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    state = session.final_state or session.initial_state
    tasks = state.get("tasks", [])

    return SessionStatusResponse(
        session_id=session.session_id,
        status=session.status,
        total_tasks=len(tasks),
        completed_tasks=len(state.get("completed_tasks", [])),
        failed_tasks=len(state.get("failed_tasks", [])),
        current_task_index=state.get("current_task_index", 0),
        model=session.model,
        hitl_level=session.hitl_level,
        has_interrupt=session.pending_interrupt is not None,
        interrupt_data=session.pending_interrupt,
    )


# ---------------------------------------------------------------------------
# Session actions
# ---------------------------------------------------------------------------

@router.post(
    "/sessions/{session_id}/run",
    response_model=SessionStatusResponse,
    dependencies=[Depends(verify_api_key)],
)
async def run_session(session_id: str, request: Request) -> SessionStatusResponse:
    """Start pipeline execution for a session."""
    manager = _get_manager(request)
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    try:
        session = await manager.start_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return SessionStatusResponse(
        session_id=session.session_id,
        status=session.status,
        model=session.model,
        hitl_level=session.hitl_level,
    )


@router.post(
    "/sessions/{session_id}/cancel",
    response_model=SessionStatusResponse,
    dependencies=[Depends(verify_api_key)],
)
async def cancel_session(session_id: str, request: Request) -> SessionStatusResponse:
    """Cancel a running or paused session."""
    manager = _get_manager(request)
    try:
        session = await manager.cancel_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return SessionStatusResponse(
        session_id=session.session_id,
        status=session.status,
    )


# ---------------------------------------------------------------------------
# Tasks & artifacts
# ---------------------------------------------------------------------------

@router.get(
    "/sessions/{session_id}/tasks",
    response_model=TaskListResponse,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def get_tasks(session_id: str, request: Request) -> TaskListResponse:
    """Get all tasks for a session."""
    manager = _get_manager(request)
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    state = session.final_state or session.initial_state
    raw_tasks = state.get("tasks", [])

    tasks = []
    for t in raw_tasks:
        if hasattr(t, "id"):
            # Pydantic Task object
            tasks.append(TaskResponse(
                id=t.id,
                title=t.title,
                description=t.description,
                task_type=t.task_type.value if hasattr(t.task_type, "value") else str(t.task_type),
                status=t.status.value if hasattr(t.status, "value") else str(t.status),
                assigned_agent=t.assigned_agent,
                retry_count=t.retry_count,
                output=t.output,
                error=t.error,
            ))
        elif isinstance(t, dict):
            tasks.append(TaskResponse(
                id=t.get("id", 0),
                title=t.get("title", ""),
                description=t.get("description", ""),
                task_type=t.get("task_type", ""),
                status=t.get("status", "pending"),
            ))

    return TaskListResponse(session_id=session_id, tasks=tasks)


@router.get(
    "/sessions/{session_id}/artifacts",
    response_model=ArtifactResponse,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def get_artifacts(session_id: str, request: Request) -> ArtifactResponse:
    """Get artifacts produced by a session."""
    manager = _get_manager(request)
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    state = session.final_state or {}
    dev_output = state.get("developer_output")
    files = []
    if dev_output:
        files = (
            dev_output.files_created
            if hasattr(dev_output, "files_created")
            else dev_output.get("files_created", [])
        )

    decisions = state.get("architect_decisions", [])
    decision_dicts = []
    for d in decisions:
        if hasattr(d, "model_dump"):
            decision_dicts.append(d.model_dump())
        elif isinstance(d, dict):
            decision_dicts.append(d)

    return ArtifactResponse(
        session_id=session_id,
        project_path=session.project_path,
        files_created=files,
        architect_decisions=decision_dicts,
    )


# ---------------------------------------------------------------------------
# Chat / HITL messages
# ---------------------------------------------------------------------------

@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatMessageResponse,
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def send_message(
    session_id: str,
    body: ChatMessageRequest,
    request: Request,
) -> ChatMessageResponse:
    """Send a message to a paused session (HITL response)."""
    manager = _get_manager(request)
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if session.status != SessionStatus.PAUSED:
        raise HTTPException(
            status_code=409,
            detail=f"Session is not paused (status={session.status.value})",
        )

    try:
        accepted = await manager.send_message(session_id, body.message)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not accepted:
        raise HTTPException(status_code=409, detail="No pending interrupt to respond to")

    return ChatMessageResponse(
        session_id=session_id,
        accepted=True,
        message="Message delivered, pipeline resuming",
        status=SessionStatus.RUNNING,
    )


# ---------------------------------------------------------------------------
# Document generation
# ---------------------------------------------------------------------------

@router.post(
    "/sessions/{session_id}/documents/generate",
    response_model=GenerateDocResponse,
    dependencies=[Depends(verify_api_key)],
)
async def generate_document(
    session_id: str,
    body: GenerateDocRequest,
    request: Request,
) -> GenerateDocResponse:
    """Generate a document for a session's project."""
    from pathlib import Path

    from eunomia_v2.documents.generator import generate_document as gen_doc

    manager = _get_manager(request)
    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    try:
        doc = await gen_doc(
            doc_type=body.doc_type,
            context=body.context,
            model=body.model,
            review=body.review,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Write to project's docs/ directory
    content = doc.to_markdown()
    file_path = ""
    if session.project_path:
        docs_dir = Path(session.project_path) / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        output_file = docs_dir / f"{body.doc_type}.md"
        output_file.write_text(content, encoding="utf-8")
        file_path = str(output_file)

    return GenerateDocResponse(
        session_id=session_id,
        doc_type=body.doc_type,
        content=content,
        file_path=file_path,
        sections_count=len(doc.sections),
    )
