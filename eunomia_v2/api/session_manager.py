"""Session lifecycle manager for the Eunomia V2 API.

Core responsibilities:
- Create/start/cancel pipeline sessions
- Buffer StreamEvents for WebSocket replay (deque with monotonic index)
- Bridge HITL interrupts between pipeline and REST/WS via asyncio.Future
- Broadcast live events to WebSocket subscribers
- Clean up expired terminal sessions
- Persist session metadata to SQLite (optional)
"""

import asyncio
import logging
import uuid
from collections import deque
from typing import Any

from eunomia_v2.api.schemas import SessionStatus
from eunomia_v2.graph.orchestrator import compile_graph
from eunomia_v2.graph.state import create_initial_state
from eunomia_v2.graph.streaming import (
    EventType,
    StreamEvent,
    resume_pipeline,
    stream_pipeline,
)
from eunomia_v2.persistence.session_store import SQLiteSessionStore

logger = logging.getLogger(__name__)

MAX_EVENT_BUFFER = 10_000


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

class Session:
    """A single pipeline execution session."""

    def __init__(
        self,
        session_id: str,
        project_path: str,
        prd_content: str,
        model: str,
        max_retries: int,
        hitl_level: str,
    ):
        self.session_id = session_id
        self.project_path = project_path
        self.prd_content = prd_content
        self.model = model
        self.max_retries = max_retries
        self.hitl_level = hitl_level
        self.status: SessionStatus = SessionStatus.CREATED

        # Graph + config — populated in start()
        self.graph: Any = None
        self.config: dict[str, Any] = {}
        self.initial_state: dict[str, Any] = {}

        # Background run task
        self.run_task: asyncio.Task | None = None

        # Event buffer — monotonic indexed
        self.event_buffer: deque[tuple[int, dict[str, Any]]] = deque(maxlen=MAX_EVENT_BUFFER)
        self._event_index: int = 0

        # WebSocket subscribers — set of asyncio.Queue
        self.subscribers: set[asyncio.Queue] = set()

        # HITL interrupt bridge
        self.pending_interrupt: dict[str, Any] | None = None
        self.interrupt_future: asyncio.Future | None = None

        # Final state snapshot
        self.final_state: dict[str, Any] = {}

    def _buffer_event(self, event: StreamEvent) -> dict[str, Any]:
        """Add an event to the buffer and return its indexed dict."""
        event_dict = event.to_dict()
        event_dict["index"] = self._event_index
        self.event_buffer.append((self._event_index, event_dict))
        self._event_index += 1
        return event_dict

    async def _broadcast(self, event_dict: dict[str, Any]) -> None:
        """Send an event dict to all active subscribers."""
        dead: list[asyncio.Queue] = []
        for q in self.subscribers:
            try:
                q.put_nowait(event_dict)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.subscribers.discard(q)


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages all active pipeline sessions."""

    def __init__(self, session_store: SQLiteSessionStore | None = None) -> None:
        self.sessions: dict[str, Session] = {}
        self.session_store = session_store or SQLiteSessionStore()

    def create_session(
        self,
        project_path: str,
        prd_content: str,
        model: str = "anthropic:claude-sonnet-4-5-20250929",
        max_retries: int = 3,
        hitl_level: str = "autonomous",
    ) -> Session:
        """Create a new pipeline session (does not start execution)."""
        session_id = uuid.uuid4().hex[:12]
        session = Session(
            session_id=session_id,
            project_path=project_path,
            prd_content=prd_content,
            model=model,
            max_retries=max_retries,
            hitl_level=hitl_level,
        )

        # Compile the LangGraph graph
        session.graph = compile_graph()
        session.config = {"configurable": {"thread_id": session_id}}
        session.initial_state = create_initial_state(
            project_path=project_path,
            prd_content=prd_content,
            model=model,
            max_retries=max_retries,
            session_id=session_id,
            hitl_level=hitl_level,
        )

        self.sessions[session_id] = session

        # Persist to SQLite
        self.session_store.save_session(
            session_id=session_id,
            status="created",
            model=model,
            project_path=project_path,
            prd_content=prd_content[:5000],
            max_retries=max_retries,
            hitl_level=hitl_level,
        )

        logger.info("Session %s created (model=%s, hitl=%s)", session_id, model, hitl_level)
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID, or None if not found."""
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[Session]:
        """Return all in-memory sessions."""
        return list(self.sessions.values())

    def list_all_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return sessions from SQLite store (includes historical sessions)."""
        return self.session_store.list_sessions(limit=limit)

    async def start_session(self, session_id: str) -> Session:
        """Start pipeline execution in a background task."""
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        if session.status not in (SessionStatus.CREATED,):
            raise ValueError(f"Session {session_id} cannot be started (status={session.status.value})")

        session.status = SessionStatus.RUNNING
        self.session_store.update_status(session_id, "running")
        session.run_task = asyncio.create_task(
            self._run_pipeline(session),
            name=f"pipeline-{session_id}",
        )
        logger.info("Session %s started", session_id)
        return session

    async def _run_pipeline(self, session: Session) -> None:
        """Background coroutine that consumes the pipeline stream.

        Buffers events, broadcasts to subscribers, and handles HITL interrupts.
        """
        try:
            async for event in stream_pipeline(
                session.graph,
                session.initial_state,
                session.config,
            ):
                event_dict = session._buffer_event(event)
                await session._broadcast(event_dict)

                # HITL interrupt — pause and wait for user input
                if event.type == EventType.INTERRUPT.value:
                    session.status = SessionStatus.PAUSED
                    session.pending_interrupt = event_dict

                    # Create a future that send_message() will resolve
                    loop = asyncio.get_running_loop()
                    session.interrupt_future = loop.create_future()

                    logger.info("Session %s paused (interrupt)", session.session_id)

                    # Block until user responds
                    user_input = await session.interrupt_future
                    session.interrupt_future = None
                    session.pending_interrupt = None
                    session.status = SessionStatus.RUNNING

                    logger.info("Session %s resumed", session.session_id)

                    # Resume pipeline with user input
                    async for resume_event in resume_pipeline(
                        session.graph,
                        user_input,
                        session.config,
                    ):
                        resume_dict = session._buffer_event(resume_event)
                        await session._broadcast(resume_dict)

                        # Handle nested interrupts
                        if resume_event.type == EventType.INTERRUPT.value:
                            session.status = SessionStatus.PAUSED
                            session.pending_interrupt = resume_dict

                            loop = asyncio.get_running_loop()
                            session.interrupt_future = loop.create_future()

                            user_input = await session.interrupt_future
                            session.interrupt_future = None
                            session.pending_interrupt = None
                            session.status = SessionStatus.RUNNING

            # Pipeline done — get final state
            try:
                state_snapshot = await session.graph.aget_state(session.config)
                session.final_state = state_snapshot.values if state_snapshot else {}
            except Exception:
                session.final_state = {}

            session.status = SessionStatus.COMPLETED
            logger.info("Session %s completed", session.session_id)

            # Persist final results
            completed = list(session.final_state.get("completed_tasks", []))
            failed = list(session.final_state.get("failed_tasks", []))
            total = len(session.final_state.get("tasks", []))
            self.session_store.update_session(
                session.session_id,
                status="completed",
                total_tasks=total,
                completed_tasks=completed,
                failed_tasks=failed,
            )

        except asyncio.CancelledError:
            session.status = SessionStatus.CANCELLED
            self.session_store.update_status(session.session_id, "cancelled")
            logger.info("Session %s cancelled", session.session_id)

        except Exception as e:
            session.status = SessionStatus.FAILED
            self.session_store.update_status(session.session_id, "failed")
            logger.exception("Session %s failed: %s", session.session_id, e)

            # Emit error event
            err_event = StreamEvent(
                event_type=EventType.ERROR,
                content=str(e),
                metadata={"exception_type": type(e).__name__},
            )
            event_dict = session._buffer_event(err_event)
            await session._broadcast(event_dict)

    async def send_message(self, session_id: str, message: str) -> bool:
        """Deliver a user message to a paused session (resolves interrupt future).

        Returns True if the message was accepted.
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        if session.interrupt_future is None or session.interrupt_future.done():
            return False

        session.interrupt_future.set_result(message)
        return True

    def subscribe(self, session_id: str, from_index: int = 0) -> asyncio.Queue:
        """Subscribe to a session's event stream.

        Returns a Queue that will receive event dicts.
        Replays buffered events from from_index first.
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        q: asyncio.Queue = asyncio.Queue(maxsize=1000)

        # Replay buffered events from the requested index
        for idx, event_dict in session.event_buffer:
            if idx >= from_index:
                try:
                    q.put_nowait(event_dict)
                except asyncio.QueueFull:
                    break

        session.subscribers.add(q)
        return q

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue from a session."""
        session = self.sessions.get(session_id)
        if session:
            session.subscribers.discard(queue)

    async def cancel_session(self, session_id: str) -> Session:
        """Cancel a running or paused session."""
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        if session.run_task and not session.run_task.done():
            session.run_task.cancel()
            try:
                await session.run_task
            except (asyncio.CancelledError, Exception):
                pass

        # Resolve any pending interrupt so the pipeline unblocks
        if session.interrupt_future and not session.interrupt_future.done():
            session.interrupt_future.cancel()

        session.status = SessionStatus.CANCELLED
        self.session_store.update_status(session_id, "cancelled")
        logger.info("Session %s cancelled", session_id)
        return session

    async def shutdown(self) -> None:
        """Cancel all running sessions — called during app shutdown."""
        for session_id in list(self.sessions):
            session = self.sessions[session_id]
            if session.run_task and not session.run_task.done():
                session.run_task.cancel()

        # Wait for all tasks to finish
        tasks = [
            s.run_task for s in self.sessions.values()
            if s.run_task and not s.run_task.done()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("SessionManager shut down (%d sessions)", len(self.sessions))

    def cleanup_expired(self, max_age_seconds: float = 3600.0) -> int:
        """Remove terminal sessions (completed/failed/cancelled) older than max_age.

        Returns the number of sessions removed.
        """
        import time

        removed = 0
        terminal = (SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED)
        now = time.time()

        for sid in list(self.sessions):
            session = self.sessions[sid]
            if session.status not in terminal:
                continue
            # Use the last event's timestamp as the session end time
            if session.event_buffer:
                _, last_event = session.event_buffer[-1]
                if now - last_event.get("timestamp", now) > max_age_seconds:
                    del self.sessions[sid]
                    removed += 1
            else:
                # No events — remove immediately
                del self.sessions[sid]
                removed += 1

        if removed:
            logger.info("Cleaned up %d expired sessions", removed)
        return removed
