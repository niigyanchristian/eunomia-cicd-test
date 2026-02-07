"""WebSocket endpoints for real-time pipeline streaming and HITL chat.

Endpoints:
- WS /api/v1/sessions/{id}/stream  — Subscribe to event stream (supports ?from=N replay)
- WS /api/v1/sessions/{id}/chat    — Bidirectional: receive events + send HITL responses
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from eunomia_v2.api.auth import verify_ws_api_key
from eunomia_v2.api.schemas import SessionStatus, WSMessage, WSMessageType
from eunomia_v2.api.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["streaming"])


def _get_manager(websocket: WebSocket) -> SessionManager:
    """Extract SessionManager from app state."""
    return websocket.app.state.session_manager


# ---------------------------------------------------------------------------
# Event stream (read-only)
# ---------------------------------------------------------------------------

@router.websocket("/sessions/{session_id}/stream")
async def stream_events(
    websocket: WebSocket,
    session_id: str,
    from_index: int = Query(default=0, alias="from"),
    _auth: str | None = Depends(verify_ws_api_key),
) -> None:
    """Stream pipeline events to the client.

    Replays buffered events from ?from=N, then streams live events.
    Sends WSMessage frames as JSON.
    """
    manager = _get_manager(websocket)
    session = manager.get_session(session_id)
    if session is None:
        await websocket.close(code=4404, reason="Session not found")
        return

    await websocket.accept()

    queue: asyncio.Queue | None = None
    try:
        queue = manager.subscribe(session_id, from_index=from_index)

        while True:
            try:
                event_dict = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                msg = WSMessage(type=WSMessageType.PONG, data={"keepalive": True})
                await websocket.send_json(msg.model_dump())
                continue

            # Determine message type
            msg_type = WSMessageType.EVENT
            if event_dict.get("type") == "interrupt":
                msg_type = WSMessageType.INTERRUPT
            elif event_dict.get("type") == "error":
                msg_type = WSMessageType.ERROR

            msg = WSMessage(
                type=msg_type,
                data=event_dict,
                index=event_dict.get("index"),
            )
            await websocket.send_json(msg.model_dump())

            # Close after pipeline ends
            if event_dict.get("type") == "pipeline_end":
                break

    except WebSocketDisconnect:
        logger.info("WS stream disconnected: session %s", session_id)
    except Exception as e:
        logger.exception("WS stream error: session %s — %s", session_id, e)
        try:
            err = WSMessage(type=WSMessageType.ERROR, data={"error": str(e)})
            await websocket.send_json(err.model_dump())
        except Exception:
            pass
    finally:
        if queue is not None:
            manager.unsubscribe(session_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Bidirectional chat (events + HITL responses)
# ---------------------------------------------------------------------------

@router.websocket("/sessions/{session_id}/chat")
async def chat_ws(
    websocket: WebSocket,
    session_id: str,
    from_index: int = Query(default=0, alias="from"),
    _auth: str | None = Depends(verify_ws_api_key),
) -> None:
    """Bidirectional WebSocket for events + HITL interaction.

    Server sends: event/interrupt/error frames.
    Client sends: {"type": "chat", "data": {"message": "..."}} to respond to interrupts,
                  or {"type": "ping"} for keepalive.
    """
    manager = _get_manager(websocket)
    session = manager.get_session(session_id)
    if session is None:
        await websocket.close(code=4404, reason="Session not found")
        return

    await websocket.accept()

    queue: asyncio.Queue | None = None
    try:
        queue = manager.subscribe(session_id, from_index=from_index)

        async def _send_events() -> None:
            """Forward events from queue to WebSocket."""
            assert queue is not None
            while True:
                try:
                    event_dict = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    msg = WSMessage(type=WSMessageType.PONG, data={"keepalive": True})
                    await websocket.send_json(msg.model_dump())
                    continue

                msg_type = WSMessageType.EVENT
                if event_dict.get("type") == "interrupt":
                    msg_type = WSMessageType.INTERRUPT
                elif event_dict.get("type") == "error":
                    msg_type = WSMessageType.ERROR

                msg = WSMessage(
                    type=msg_type,
                    data=event_dict,
                    index=event_dict.get("index"),
                )
                await websocket.send_json(msg.model_dump())

                if event_dict.get("type") == "pipeline_end":
                    return

        async def _receive_messages() -> None:
            """Receive client messages and handle HITL responses."""
            while True:
                raw = await websocket.receive_text()
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    err = WSMessage(type=WSMessageType.ERROR, data={"error": "Invalid JSON"})
                    await websocket.send_json(err.model_dump())
                    continue

                msg_type = payload.get("type", "")

                if msg_type == "ping":
                    pong = WSMessage(type=WSMessageType.PONG, data={})
                    await websocket.send_json(pong.model_dump())

                elif msg_type == "chat":
                    message = payload.get("data", {}).get("message", "")
                    if not message:
                        err = WSMessage(
                            type=WSMessageType.ERROR,
                            data={"error": "Empty message"},
                        )
                        await websocket.send_json(err.model_dump())
                        continue

                    try:
                        accepted = await manager.send_message(session_id, message)
                        if not accepted:
                            err = WSMessage(
                                type=WSMessageType.ERROR,
                                data={"error": "No pending interrupt"},
                            )
                            await websocket.send_json(err.model_dump())
                    except ValueError as e:
                        err = WSMessage(
                            type=WSMessageType.ERROR,
                            data={"error": str(e)},
                        )
                        await websocket.send_json(err.model_dump())

                else:
                    err = WSMessage(
                        type=WSMessageType.ERROR,
                        data={"error": f"Unknown message type: {msg_type}"},
                    )
                    await websocket.send_json(err.model_dump())

        # Run sender and receiver concurrently
        sender = asyncio.create_task(_send_events())
        receiver = asyncio.create_task(_receive_messages())

        done, pending = await asyncio.wait(
            {sender, receiver},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    except WebSocketDisconnect:
        logger.info("WS chat disconnected: session %s", session_id)
    except Exception as e:
        logger.exception("WS chat error: session %s — %s", session_id, e)
    finally:
        if queue is not None:
            manager.unsubscribe(session_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
