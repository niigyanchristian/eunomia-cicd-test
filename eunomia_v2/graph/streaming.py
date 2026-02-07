"""Stream event formatting and adapters for CLI and WebSocket output.

Processes LangGraph astream() updates into structured StreamEvents
for real-time CLI display and WebSocket transmission.

Stream modes supported:
- "updates": {node_name: state_update} dicts — primary mode
- "messages": (message_chunk, metadata) tuples — token-level

HITL support (M8):
- Detects __interrupt__ events from LangGraph interrupt() calls
- resume_pipeline() resumes after interrupt with Command(resume=...)
"""

import time
from enum import Enum
from typing import Any, AsyncIterator


class EventType(str, Enum):
    """Structured event types emitted during pipeline execution."""

    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    TOKEN = "token"
    TOOL_CALL = "tool_call"
    STATE_UPDATE = "state_update"
    INTERRUPT = "interrupt"
    ERROR = "error"


class StreamEvent:
    """A structured event emitted during graph execution."""

    def __init__(
        self,
        event_type: "EventType | str",
        agent: str = "",
        content: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        self.type = event_type.value if isinstance(event_type, EventType) else event_type
        self.agent = agent
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for WebSocket / JSON transmission."""
        return {
            "type": self.type,
            "agent": self.agent,
            "content": self.content,
            "timestamp": self.timestamp,
            **self.metadata,
        }

    def __repr__(self) -> str:
        return f"StreamEvent(type={self.type!r}, agent={self.agent!r}, content={self.content!r})"


# Agent display names for terminal output
AGENT_DISPLAY: dict[str, str] = {
    "planner": "Planner",
    "task_router": "Task Router",
    "developer": "Developer",
    "qa": "QA",
    "devops": "DevOps",
    "architect": "Architect",
    "commit": "Commit",
    "escalate": "Escalate",
}


def format_stream_event(langgraph_event: Any) -> dict[str, Any]:
    """Convert a raw LangGraph stream event into our StreamEvent format.

    LangGraph emits different event shapes depending on stream_mode:
    - "messages": (message_chunk, metadata) tuples
    - "updates": {node_name: state_update} dicts
    - "values": full state snapshots

    Args:
        langgraph_event: Raw event from graph.astream()

    Returns:
        Serialized StreamEvent dict
    """
    # Handle (message_chunk, metadata) tuples from stream_mode="messages"
    if isinstance(langgraph_event, tuple) and len(langgraph_event) == 2:
        chunk, metadata = langgraph_event
        node = metadata.get("langgraph_node", "unknown")
        content = getattr(chunk, "content", str(chunk))
        return StreamEvent(
            event_type=EventType.TOKEN,
            agent=node,
            content=content,
        ).to_dict()

    # Handle {node_name: update} dicts from stream_mode="updates"
    if isinstance(langgraph_event, dict):
        for node_name, update in langgraph_event.items():
            return StreamEvent(
                event_type=EventType.STATE_UPDATE,
                agent=node_name,
                content="",
                metadata={"update": update} if isinstance(update, dict) else {},
            ).to_dict()

    # Fallback
    return StreamEvent(
        event_type="unknown",
        content=str(langgraph_event),
    ).to_dict()


# ---------------------------------------------------------------------------
# Shared state tracker for stream processing
# ---------------------------------------------------------------------------

class _StreamState:
    """Mutable tracking state shared across stream processing calls."""

    def __init__(self) -> None:
        self.tasks: list = []
        self.completed: set[str] = set()
        self.failed: set[str] = set()
        self.active_node: str = ""


def _parse_interrupt_value(raw: Any) -> dict[str, Any]:
    """Extract interrupt info from a LangGraph Interrupt object or raw value."""
    value = raw.value if hasattr(raw, "value") else raw
    if isinstance(value, dict):
        return value
    return {"message": str(value)}


async def _process_updates(
    astream_iter: Any,
    ss: _StreamState,
) -> AsyncIterator[StreamEvent]:
    """Process a stream of LangGraph updates into StreamEvents.

    Shared logic used by both stream_pipeline() and resume_pipeline().
    Yields StreamEvent objects and updates ss (shared state) in place.
    """
    async for update in astream_iter:
        if not isinstance(update, dict):
            continue

        # Detect __interrupt__ events from LangGraph interrupt() calls
        if "__interrupt__" in update:
            interrupt_data = update["__interrupt__"]
            # interrupt_data is a tuple/list of Interrupt objects
            items = interrupt_data if isinstance(interrupt_data, (list, tuple)) else [interrupt_data]
            for intr in items:
                iv = _parse_interrupt_value(intr)
                yield StreamEvent(
                    event_type=EventType.INTERRUPT,
                    agent=ss.active_node,
                    content=iv.get("message", str(iv)),
                    metadata={
                        "interrupt_type": iv.get("type", "unknown"),
                        "interrupt_data": iv,
                    },
                )
            # Stream pauses after interrupt — caller must resume
            continue

        for node_name, state_update in update.items():
            # Emit agent transitions
            if node_name != ss.active_node:
                if ss.active_node:
                    yield StreamEvent(
                        event_type=EventType.AGENT_END,
                        agent=ss.active_node,
                        content=f"{AGENT_DISPLAY.get(ss.active_node, ss.active_node)} finished",
                    )
                ss.active_node = node_name
                yield StreamEvent(
                    event_type=EventType.AGENT_START,
                    agent=node_name,
                    content=f"{AGENT_DISPLAY.get(node_name, node_name)} started",
                )

            if not isinstance(state_update, dict):
                continue

            # Track tasks list from planner
            if "tasks" in state_update:
                ss.tasks = state_update["tasks"]
                yield StreamEvent(
                    event_type=EventType.STATE_UPDATE,
                    agent=node_name,
                    content=f"Generated {len(ss.tasks)} tasks",
                    metadata={"total_tasks": len(ss.tasks)},
                )

            # Track task routing
            idx = state_update.get("current_task_index")
            if idx is not None and ss.tasks and idx < len(ss.tasks) and node_name == "task_router":
                task = ss.tasks[idx]
                tid = task.id if hasattr(task, "id") else task.get("id")
                title = task.title if hasattr(task, "title") else task.get("title", "")
                tt = task.task_type if hasattr(task, "task_type") else task.get("task_type", "")
                tt_str = tt.value if hasattr(tt, "value") else str(tt)
                yield StreamEvent(
                    event_type=EventType.TASK_START,
                    agent=node_name,
                    content=f"Task [{tid}] {title}",
                    metadata={
                        "task_id": tid,
                        "task_title": title,
                        "task_type": tt_str,
                    },
                )

            # Track completions
            if "completed_tasks" in state_update:
                new_completed = set(str(t) for t in state_update["completed_tasks"])
                newly_done = new_completed - ss.completed
                ss.completed = new_completed
                for tid in newly_done:
                    yield StreamEvent(
                        event_type=EventType.TASK_COMPLETE,
                        agent=node_name,
                        content=f"Task [{tid}] completed",
                        metadata={
                            "task_id": tid,
                            "completed_count": len(ss.completed),
                            "total_tasks": len(ss.tasks),
                        },
                    )

            # Track failures
            if "failed_tasks" in state_update:
                new_failed = set(str(t) for t in state_update["failed_tasks"])
                newly_failed = new_failed - ss.failed
                ss.failed = new_failed
                for tid in newly_failed:
                    yield StreamEvent(
                        event_type=EventType.TASK_FAILED,
                        agent=node_name,
                        content=f"Task [{tid}] failed",
                        metadata={
                            "task_id": tid,
                            "failed_count": len(ss.failed),
                        },
                    )

            # Track QA results
            qa_result = state_update.get("qa_result")
            if qa_result is not None:
                passed = qa_result.passed if hasattr(qa_result, "passed") else qa_result.get("passed")
                yield StreamEvent(
                    event_type=EventType.STATE_UPDATE,
                    agent=node_name,
                    content=f"Tests {'PASSED' if passed else 'FAILED'}",
                    metadata={"qa_passed": passed},
                )

            # Track developer output
            dev_output = state_update.get("developer_output")
            if dev_output is not None:
                success = dev_output.success if hasattr(dev_output, "success") else dev_output.get("success")
                files = dev_output.files_created if hasattr(dev_output, "files_created") else dev_output.get("files_created", [])
                yield StreamEvent(
                    event_type=EventType.STATE_UPDATE,
                    agent=node_name,
                    content=f"{'Success' if success else 'Failed'} -- {len(files)} files",
                    metadata={"success": success, "files_created": len(files)},
                )

            # Track retry count
            if "retry_count" in state_update and state_update["retry_count"] > 0:
                yield StreamEvent(
                    event_type=EventType.STATE_UPDATE,
                    agent=node_name,
                    content=f"Retry #{state_update['retry_count']}",
                    metadata={"retry_count": state_update["retry_count"]},
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def stream_pipeline(
    graph: Any,
    initial_state: dict[str, Any],
    config: dict[str, Any],
) -> AsyncIterator[StreamEvent]:
    """Stream pipeline execution as structured events.

    Wraps graph.astream() with stream_mode="updates" and converts
    raw LangGraph updates into structured StreamEvent objects.

    When a node calls interrupt(), yields an INTERRUPT event and the
    stream ends. The caller should prompt the user, then call
    resume_pipeline() to continue.

    Yields:
        StreamEvent objects for each graph transition.
    """
    yield StreamEvent(
        event_type=EventType.PIPELINE_START,
        content="Pipeline execution started",
        metadata={"session_id": initial_state.get("session_id", "")},
    )

    ss = _StreamState()

    try:
        astream = graph.astream(initial_state, config=config, stream_mode="updates")
        async for event in _process_updates(astream, ss):
            yield event

    except Exception as e:
        yield StreamEvent(
            event_type=EventType.ERROR,
            content=str(e),
            metadata={"exception_type": type(e).__name__},
        )
        raise

    finally:
        if ss.active_node:
            yield StreamEvent(
                event_type=EventType.AGENT_END,
                agent=ss.active_node,
                content=f"{AGENT_DISPLAY.get(ss.active_node, ss.active_node)} finished",
            )

        yield StreamEvent(
            event_type=EventType.PIPELINE_END,
            content="Pipeline execution completed",
            metadata={
                "completed_count": len(ss.completed),
                "failed_count": len(ss.failed),
                "total_tasks": len(ss.tasks),
            },
        )


async def resume_pipeline(
    graph: Any,
    resume_value: Any,
    config: dict[str, Any],
) -> AsyncIterator[StreamEvent]:
    """Resume a paused pipeline with user input after an interrupt.

    Sends Command(resume=resume_value) to the graph and continues
    streaming events. If another interrupt occurs, yields another
    INTERRUPT event and the stream ends again.

    Yields:
        StreamEvent objects for the resumed execution.
    """
    from langgraph.types import Command

    yield StreamEvent(
        event_type=EventType.STATE_UPDATE,
        content="Pipeline resumed with user input",
        metadata={"resume_value": str(resume_value)[:200]},
    )

    ss = _StreamState()

    try:
        astream = graph.astream(
            Command(resume=resume_value), config=config, stream_mode="updates"
        )
        async for event in _process_updates(astream, ss):
            yield event

    except Exception as e:
        yield StreamEvent(
            event_type=EventType.ERROR,
            content=str(e),
            metadata={"exception_type": type(e).__name__},
        )
        raise

    finally:
        if ss.active_node:
            yield StreamEvent(
                event_type=EventType.AGENT_END,
                agent=ss.active_node,
                content=f"{AGENT_DISPLAY.get(ss.active_node, ss.active_node)} finished",
            )

        yield StreamEvent(
            event_type=EventType.PIPELINE_END,
            content="Pipeline execution completed",
            metadata={
                "completed_count": len(ss.completed),
                "failed_count": len(ss.failed),
                "total_tasks": len(ss.tasks),
            },
        )
