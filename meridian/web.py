"""Meridian Web Server — FastAPI + WebSocket, backed by claude CLI.

Spawns `claude -p` subprocesses for each message, streams output
over WebSocket to the browser chat UI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from meridian.config import config
from meridian.gateway import MemoryGateway
from meridian.storage import StorageLayer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("meridian.web")

# ── Global state ──
storage: StorageLayer | None = None
gateway: MemoryGateway | None = None
sandbox_dir: Path | None = None  # Safe temp dir for claude -p subprocesses

# Map WebSocket connections to their Claude session IDs (for --resume)
ws_sessions: dict[int, str] = {}

# Meridian project root (for .mcp.json reference)
MERIDIAN_ROOT = Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    global storage, gateway, sandbox_dir

    logger.info("=== Meridian Web Server starting ===")

    # Check claude CLI is available
    claude_path = shutil.which("claude")
    if not claude_path:
        logger.error("claude CLI not found in PATH! Install Claude Code first.")
    else:
        logger.info(f"Claude CLI: {claude_path}")

    storage = StorageLayer()
    await storage.init_db()
    gateway = MemoryGateway()

    # Create an isolated sandbox dir for claude -p subprocesses.
    # This prevents Claude from reading/modifying Meridian's own source.
    # We copy .mcp.json there so it can discover the MCP server.
    import tempfile
    sandbox_dir = Path(tempfile.mkdtemp(prefix="meridian_sandbox_"))
    mcp_src = MERIDIAN_ROOT / ".mcp.json"
    mcp_dst = sandbox_dir / ".mcp.json"
    if mcp_src.exists():
        import shutil as sh
        sh.copy2(mcp_src, mcp_dst)
    logger.info(f"Sandbox dir: {sandbox_dir}")
    logger.info(f"Web UI: http://localhost:{config.port}")

    # Load channel history from SQLite
    _init_channel_db()

    yield

    if storage:
        await storage.close()
    if gateway:
        await gateway.close()
    # Clean up sandbox
    if sandbox_dir and sandbox_dir.exists():
        import shutil as sh
        sh.rmtree(sandbox_dir, ignore_errors=True)


app = FastAPI(title="Meridian", version="0.3.0", lifespan=lifespan)

# CORS: allow browser artifacts and external consumers to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Routes ──

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse("<h1>Meridian</h1><p>index.html not found</p>", status_code=404)


@app.get("/api/health")
async def health():
    qdrant_ok = False
    try:
        storage.qdrant.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{config.ollama_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    claude_ok = shutil.which("claude") is not None

    return {
        "status": "ok" if (qdrant_ok and ollama_ok and claude_ok) else "degraded",
        "qdrant": "ok" if qdrant_ok else "down",
        "ollama": "ok" if ollama_ok else "down",
        "claude_cli": "ok" if claude_ok else "missing",
        "active_sessions": len(ws_sessions),
    }


@app.get("/api/projects")
async def list_projects():
    projects = await storage.list_projects()
    pids = {p["project_id"] for p in projects}
    if config.project_id not in pids:
        projects.insert(0, {"project_id": config.project_id})
    return {"projects": projects}


@app.get("/api/projects/{project_id}/status")
async def project_status(project_id: str):
    return await storage.project_status(project_id)


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": [{"ws_id": k, "claude_session": v} for k, v in ws_sessions.items()]}


# ── Memory REST API (for external consumers and inter-agent communication) ──


class RecallRequest(BaseModel):
    query: str
    scope: str = "all"
    max_tokens: int = 800


class RememberRequest(BaseModel):
    content: str
    type: str = "note"
    tags: list[str] = []
    importance: int = 3
    source: str | None = None


class CheckpointRequest(BaseModel):
    task_state: str
    decisions: list[str] = []
    warnings: list[str] = []
    next_steps: list[str] = []
    working_set: dict | str = ""


class MessageRequest(BaseModel):
    """Inter-Claude message: one Claude sends a message, the other polls for it."""
    from_id: str
    to_id: str
    content: str
    msg_type: str = "text"  # text, task, result, query


# Simple in-memory message queue for Claude-to-Claude comms
_message_queue: list[dict] = []

# ── IRC Channel ──
_channel_messages: list[dict] = []
_channel_subscribers: list[WebSocket] = []
_channel_db_path = Path(os.getenv("MERIDIAN_DATA_DIR", str(Path.home() / ".meridian"))) / "channel.db"


@app.post("/api/memory/recall")
async def api_recall(req: RecallRequest):
    """Semantic memory recall — available to external consumers via REST."""
    result = await gateway.recall(req.query, req.scope, req.max_tokens, storage)
    return result


@app.post("/api/memory/remember")
async def api_remember(req: RememberRequest):
    """Store a new memory."""
    from meridian.workers import WorkerPool
    workers = WorkerPool()
    try:
        result = await storage.remember(
            {
                "content": req.content,
                "type": req.type,
                "tags": req.tags,
                "importance": req.importance,
                "source": req.source,
                "related_to": [],
            },
            workers,
        )
        return result
    finally:
        await workers.close()


@app.get("/api/memory/briefing")
async def api_briefing(project_id: str = "default"):
    """Get the current hot memory briefing."""
    briefing = await gateway.assemble_briefing(project_id, storage)
    return {"briefing": briefing}


@app.post("/api/checkpoint")
async def api_checkpoint(req: CheckpointRequest):
    """Save a session checkpoint. Used by precompact hook and CLI."""
    working_set = req.working_set if isinstance(req.working_set, dict) else {}
    result = await storage.checkpoint(
        {
            "task_state": req.task_state,
            "decisions": req.decisions,
            "warnings": req.warnings,
            "next_steps": req.next_steps,
            "working_set": working_set,
        }
    )
    return result


@app.post("/api/bridge/send")
async def bridge_send(req: MessageRequest):
    """Send a message from one Claude to another."""
    import time
    msg = {
        "from": req.from_id,
        "to": req.to_id,
        "content": req.content,
        "type": req.msg_type,
        "ts": time.time(),
    }
    _message_queue.append(msg)
    logger.info(f"Bridge msg: {req.from_id} → {req.to_id} ({req.msg_type})")
    return {"queued": True, "queue_depth": len(_message_queue)}


@app.get("/api/bridge/poll")
async def bridge_poll(recipient: str, since: float = 0):
    """Poll for messages addressed to a specific Claude."""
    msgs = [m for m in _message_queue if m["to"] == recipient and m["ts"] > since]
    return {"messages": msgs, "count": len(msgs)}


@app.post("/api/bridge/ack")
async def bridge_ack(recipient: str, before: float = 0):
    """Acknowledge (clear) messages older than a timestamp."""
    global _message_queue
    before_len = len(_message_queue)
    _message_queue = [
        m for m in _message_queue
        if not (m["to"] == recipient and m["ts"] <= before)
    ]
    cleared = before_len - len(_message_queue)
    return {"cleared": cleared, "remaining": len(_message_queue)}


# ── IRC Channel ──

class ChannelMessage(BaseModel):
    sender: str
    content: str


def _init_channel_db():
    """Initialize channel SQLite and load recent messages into memory."""
    import sqlite3
    db = sqlite3.connect(str(_channel_db_path))
    db.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sender TEXT NOT NULL,
        content TEXT NOT NULL,
        ts REAL NOT NULL
    )""")
    db.commit()
    # Load last 200 messages into memory
    rows = db.execute("SELECT sender, content, ts FROM messages ORDER BY id DESC LIMIT 200").fetchall()
    for sender, content, ts in reversed(rows):
        _channel_messages.append({"sender": sender, "content": content, "ts": ts})
    db.close()
    logger.info(f"Channel DB: {_channel_db_path} ({len(_channel_messages)} messages loaded)")


async def _channel_broadcast(sender: str, content: str) -> dict:
    """Store a message, broadcast to subscribers, return the entry."""
    import sqlite3
    ts = time.time()
    entry = {"sender": sender, "content": content, "ts": ts}

    # Persist to SQLite
    try:
        db = sqlite3.connect(str(_channel_db_path))
        db.execute("INSERT INTO messages (sender, content, ts) VALUES (?, ?, ?)", (sender, content, ts))
        db.commit()
        db.close()
    except Exception as e:
        logger.warning(f"Channel DB write failed: {e}")

    # In-memory cache
    _channel_messages.append(entry)
    if len(_channel_messages) > 500:
        _channel_messages[:] = _channel_messages[-500:]

    # Broadcast to WebSocket subscribers
    dead = []
    for sub in _channel_subscribers:
        try:
            await sub.send_json(entry)
        except Exception:
            dead.append(sub)
    for d in dead:
        _channel_subscribers.remove(d)

    # Forward to bridge message queue so bridge processes (bridge.py,
    # meridian_bridge.py) can deliver to Webbie via executor.
    # Skip if sender is webbie to avoid echo loops.
    if sender.lower() != "webbie":
        bridge_msg = {
            "from": sender,
            "to": "webbie",
            "content": f"[#{sender}] {content}",
            "type": "channel",
            "ts": ts,
        }
        _message_queue.append(bridge_msg)
        logger.info(f"[channel->bridge] Forwarded to bridge queue for webbie")

    logger.info(f"[channel] {sender}: {content[:80]}")
    return entry


@app.get("/channel", response_class=HTMLResponse)
async def channel_page():
    channel_path = static_dir / "channel.html"
    if channel_path.exists():
        return HTMLResponse(content=channel_path.read_text())
    return HTMLResponse("<h1>channel.html not found</h1>", status_code=404)


@app.post("/api/channel/send")
async def channel_send(msg: ChannelMessage):
    """Post a message to the shared channel. Everyone sees it."""
    await _channel_broadcast(msg.sender, msg.content)
    return {"ok": True, "count": len(_channel_messages)}


@app.get("/api/channel/say")
async def channel_say(nick: str, msg: str):
    """Simple GET endpoint. Usage: /api/channel/say?nick=chris&msg=hello+world"""
    await _channel_broadcast(nick, msg)
    return {"ok": True}


@app.get("/api/channel/history")
async def channel_history(since: float = 0, limit: int = 100):
    """Get recent channel messages."""
    msgs = [m for m in _channel_messages if m["ts"] > since]
    return {"messages": msgs[-limit:]}


@app.websocket("/ws/channel")
async def channel_ws(ws: WebSocket):
    """WebSocket for real-time channel updates."""
    await ws.accept()
    _channel_subscribers.append(ws)
    logger.info(f"Channel subscriber connected ({len(_channel_subscribers)} total)")
    try:
        while True:
            # Client can send messages through the WebSocket too
            data = await ws.receive_json()
            if "sender" in data and "content" in data:
                await _channel_broadcast(data["sender"], data["content"])
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Channel WS error: {e}")
    finally:
        if ws in _channel_subscribers:
            _channel_subscribers.remove(ws)
        logger.info(f"Channel subscriber disconnected ({len(_channel_subscribers)} total)")


# ── WebSocket Chat ──

@app.websocket("/ws/{project_id}")
async def websocket_chat(ws: WebSocket, project_id: str = "default"):
    """DISABLED — was spawning rogue claude -p subprocesses (zombie factory).
    Use claude-launcher.py for the real session, not this endpoint."""
    await ws.accept()
    await ws.send_json({"type": "error", "data": "WebSocket chat disabled — use claude-launcher.py instead"})
    await ws.close()
    return
    # --- ORIGINAL CODE BELOW (disabled) ---
    ws_id = id(ws)
    logger.info(f"WebSocket connected: {ws_id} for project {project_id}")

    storage.project_id = project_id

    # Assemble briefing for the system prompt appendix
    briefing = await gateway.assemble_briefing(project_id, storage)
    system_appendix = _build_system_appendix(project_id, briefing)

    await ws.send_json({"type": "briefing", "data": system_appendix})
    await ws.send_json({"type": "ready", "data": {"session_id": f"ws_{ws_id}"}})

    try:
        while True:
            user_input = await ws.receive_text()
            logger.info(f"[{ws_id}] User: {user_input[:100]}...")

            # Build claude CLI command
            mcp_config_path = str(sandbox_dir / ".mcp.json")
            cmd = [
                "claude", "-p", user_input,
                "--output-format", "stream-json", "--verbose", # WETWARE: --verbose is required when using print w/ output-format
                "--dangerously-skip-permissions",
                "--mcp-config", mcp_config_path,
                "--append-system-prompt", system_appendix,
            ]

            # Resume session if we have one
            session_id = ws_sessions.get(ws_id)
            if session_id:
                cmd.extend(["--resume", session_id])

            # Spawn claude subprocess in the sandbox dir (not Meridian's own source tree).
            # Unset CLAUDECODE to avoid nesting check when launched from another Claude session.
            env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(sandbox_dir),
                    env=env,
                )
            except FileNotFoundError:
                await ws.send_json({"type": "error", "data": "claude CLI not found. Run: npm install -g @anthropic-ai/claude-code"})
                continue

            # Stream stdout → WebSocket
            accumulated_text = ""
            async for line in proc.stdout:
                line_str = line.decode().strip()
                if not line_str:
                    continue

                try:
                    event = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                # Text content from assistant
                if event_type == "assistant":
                    content_blocks = event.get("message", {}).get("content", [])
                    for block in content_blocks:
                        if block.get("type") == "text":
                            text = block["text"]
                            # Only send new text (delta)
                            if text.startswith(accumulated_text):
                                delta = text[len(accumulated_text):]
                            else:
                                delta = text
                            accumulated_text = text
                            if delta:
                                await ws.send_json({"type": "content", "data": delta})

                # Tool calls
                elif event_type == "tool_use":
                    tool_info = event.get("tool", {})
                    await ws.send_json({
                        "type": "tool_call",
                        "data": {"name": tool_info.get("name", "unknown"), "input": tool_info.get("input", {})},
                    })

                # Tool results
                elif event_type == "tool_result":
                    tool_content = str(event.get("content", ""))
                    await ws.send_json({
                        "type": "tool_result",
                        "data": {"preview": tool_content[:200]},
                    })

                # Final result — capture session ID for --resume
                elif event_type == "result":
                    new_session_id = event.get("session_id")
                    if new_session_id:
                        ws_sessions[ws_id] = new_session_id
                        logger.info(f"[{ws_id}] Claude session: {new_session_id}")

                    # Send final text if we haven't already
                    result_text = event.get("result", "")
                    if result_text and result_text != accumulated_text:
                        delta = result_text[len(accumulated_text):] if result_text.startswith(accumulated_text) else result_text
                        if delta:
                            await ws.send_json({"type": "content", "data": delta})

            # Wait for process to finish
            stderr_data = await proc.stderr.read()
            await proc.wait()

            if proc.returncode != 0 and stderr_data:
                error_msg = stderr_data.decode().strip()
                if error_msg:
                    logger.warning(f"[{ws_id}] claude stderr: {error_msg[:200]}")
                    # Only send as error if it's actually an error, not just info/warnings
                    if "error" in error_msg.lower() or proc.returncode != 0:
                        await ws.send_json({"type": "error", "data": error_msg[:500]})

            await ws.send_json({"type": "done"})

            # Refresh briefing periodically (every 5th message or so)
            # The append-system-prompt will include the latest state
            briefing = await gateway.assemble_briefing(project_id, storage)
            system_appendix = _build_system_appendix(project_id, briefing)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {ws_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass
    finally:
        ws_sessions.pop(ws_id, None)


def _build_system_appendix(project_id: str, briefing: str) -> str:
    """Build the system prompt appendix injected via --append-system-prompt."""
    return f"""You have persistent memory via the Meridian MCP server. USE IT ACTIVELY.

MEMORY PROTOCOL:
- Use memory_recall when you need context about past decisions, architecture, or history.
- Use memory_remember when you make decisions, find bugs, establish patterns, or learn something important.
- Use memory_checkpoint before ANY compaction or when wrapping up work. NON-NEGOTIABLE.
- When you discover a gotcha or footgun, store it as a warning immediately.
- Keep your context lean. Query for what you need, when you need it.

CURRENT BRIEFING (project: {project_id}):
{briefing}"""


if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
