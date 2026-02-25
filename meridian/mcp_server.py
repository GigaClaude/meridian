"""Meridian MCP Server — standalone FastMCP server for Claude Code.

Runs as a separate process. Claude Code connects via stdio transport.

Usage:
    python -m meridian.mcp_server
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

# Need to set up path for when run as __main__
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from meridian.config import config
from meridian.email import EmailClient
from meridian.gateway import MemoryGateway
from meridian.storage import StorageLayer
from meridian.workers import WorkerPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],  # stderr so stdout stays clean for MCP protocol
)
logger = logging.getLogger("meridian.mcp")

# ── Subsystems (initialized on startup) ──
storage: StorageLayer | None = None
gateway: MemoryGateway | None = None
workers: WorkerPool | None = None
email_client: EmailClient | None = None


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Initialize subsystems when MCP server starts."""
    global storage, gateway, workers, email_client
    logger.info("Meridian MCP server starting...")
    storage = StorageLayer()
    await storage.init_db()
    gateway = MemoryGateway()
    workers = WorkerPool()
    email_client = EmailClient()
    if config.gmail_address:
        logger.info(f"Email configured: {config.gmail_address}")
    else:
        logger.warning("Email not configured — set GMAIL_ADDRESS and GMAIL_TOKEN_PATH")
    logger.info("Meridian MCP server ready — all subsystems initialized")
    yield
    logger.info("Meridian MCP server shutting down...")
    if storage:
        await storage.close()
    if gateway:
        await gateway.close()
    if workers:
        await workers.close()
    if email_client:
        await email_client.close()


mcp = FastMCP(
    "meridian",
    instructions=(
        "Meridian provides persistent memory across sessions. "
        "Use memory_recall to search past decisions, patterns, and debugging history. "
        "Use memory_remember to store important decisions, warnings, and patterns. "
        "Use memory_checkpoint before compaction to preserve session state."
    ),
    lifespan=lifespan,
)


# ── Memory Tools ──

@mcp.tool()
async def memory_briefing(project_id: str = "default") -> str:
    """Get hot memory briefing for current project. Returns YAML with current task state, active decisions, warnings, and working set. Call at session start or after compaction."""
    storage.project_id = project_id
    briefing = await gateway.assemble_briefing(project_id, storage)
    token_count = len(briefing.split())
    return f"[{token_count} tokens]\n{briefing}"


@mcp.tool()
async def memory_recall(query: str, scope: str = "all", max_tokens: int = 800) -> str:
    """Search warm memory for relevant context. Use when you need information about past decisions, architecture, debugging history, or patterns. Returns a synthesized summary.

    Args:
        query: Natural language query describing what you need to know
        scope: Filter by memory type — decisions, patterns, debug, entities, warnings, or all
        max_tokens: Maximum response size in tokens (default: 800)
    """
    result = await gateway.recall(query, scope, max_tokens, storage)
    sources_str = ""
    if result.get("sources"):
        sources_str = "\n\nSources: " + ", ".join(
            f"[{s['id']}]({s['type']}, relevance={s.get('relevance', 0):.2f})"
            for s in result["sources"]
        )
    return result.get("results", "No results found.") + sources_str


@mcp.tool()
async def memory_remember(
    content: str,
    type: str,
    tags: list[str] | None = None,
    importance: int = 3,
    related_to: list[str] | None = None,
    source: str | None = None,
) -> str:
    """Store new knowledge into persistent memory. Use when you make decisions, find bugs, establish patterns, or learn something important that should survive across sessions.

    Args:
        content: What to remember — be specific and include rationale
        type: Category — decision, pattern, debug, entity, warning, or note
        tags: Searchable tags for this memory
        importance: Retention priority 1-5 (1=low, 5=critical, default=3)
        related_to: IDs of related memories
        source: Attribution — who created this memory (e.g. giga, webbie, chris)
    """
    result = await storage.remember(
        {
            "content": content,
            "type": type,
            "tags": tags or [],
            "importance": importance,
            "source": source,
            "related_to": related_to or [],
        },
        workers,
    )
    return json.dumps(result, default=str)


@mcp.tool()
async def memory_checkpoint(
    task_state: str,
    decisions: list[str] | None = None,
    warnings: list[str] | None = None,
    next_steps: list[str] | None = None,
    working_set: dict | None = None,
) -> str:
    """Save current session state before compaction or session end. CRITICAL: Call this before any compaction event to preserve continuity.

    Args:
        task_state: What you're currently working on
        decisions: Decisions made this session
        warnings: New gotchas discovered
        next_steps: What to do next session
        working_set: Dict with 'files' and 'endpoints' lists
    """
    result = await storage.checkpoint(
        {
            "task_state": task_state,
            "decisions": decisions or [],
            "warnings": warnings or [],
            "next_steps": next_steps or [],
            "working_set": working_set or {},
        }
    )
    return json.dumps(result, default=str)


@mcp.tool()
async def memory_graph_query(entity: str, relation: str | None = None, depth: int = 2) -> str:
    """Traverse the entity relationship graph. Use to understand how components connect — 'what calls what', 'what depends on what', etc.

    Args:
        entity: Starting entity name to traverse from
        relation: Filter by relation type (calls, depends_on, configured_by, implements, broke)
        depth: Traversal depth 1-5 (default: 2)
    """
    result = await gateway.graph_query(
        {"entity": entity, "relation": relation, "depth": depth},
        storage,
    )
    return json.dumps(result, default=str)


@mcp.tool()
async def memory_history(query: str) -> str:
    """Search episodic cold storage for specific past events. Use when you need to find something from a previous session's transcript.

    Args:
        query: What to search for in past sessions
    """
    result = await gateway.history_search({"query": query}, storage, workers)
    return json.dumps(result, default=str)


@mcp.tool()
async def memory_forget(id: str, reason: str = "") -> str:
    """Remove or deprecate a memory. Use when information is wrong or outdated.

    Args:
        id: Memory ID to forget
        reason: Why this memory is being forgotten (stored in audit log)
    """
    result = await storage.forget({"id": id, "reason": reason})
    return json.dumps(result, default=str)


@mcp.tool()
async def memory_update(id: str, content: str | None = None, tags: list[str] | None = None, importance: int | None = None, superseded_by: str | None = None) -> str:
    """Modify an existing memory. Use when a decision is reversed, a pattern changes, or information needs correction.

    Args:
        id: Memory ID to update
        content: New content (optional)
        tags: New tags (optional)
        importance: New importance 1-5 (optional)
        superseded_by: ID of memory that replaces this one (optional)
    """
    updates = {}
    if content is not None:
        updates["content"] = content
    if tags is not None:
        updates["tags"] = tags
    if importance is not None:
        updates["importance"] = importance
    if superseded_by is not None:
        updates["superseded_by"] = superseded_by

    result = await storage.update({"id": id, "updates": updates})
    return json.dumps(result, default=str)


# ── Email Tools ──

@mcp.tool()
async def email_send(to: str, subject: str, body: str, cc: str | None = None) -> str:
    """Send an email from GigaClaude's Gmail account.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body (plain text)
        cc: Optional CC addresses (comma-separated)
    """
    result = await email_client.send(to, subject, body, cc)
    return json.dumps(result, default=str)


@mcp.tool()
async def email_inbox(limit: int = 10, unread_only: bool = True) -> str:
    """Check GigaClaude's email inbox. Returns recent messages with sender, subject, date, and UID.

    Args:
        limit: Maximum number of emails to return (default: 10)
        unread_only: Only show unread messages (default: True)
    """
    messages = await email_client.fetch_inbox(limit=limit, unread_only=unread_only)
    return json.dumps(messages, default=str)


@mcp.tool()
async def email_read(uid: str) -> str:
    """Read the full content of an email by its UID (from email_inbox results).

    Args:
        uid: The email UID to read
    """
    result = await email_client.read_email(uid)
    return json.dumps(result, default=str)


if __name__ == "__main__":
    mcp.run(transport="stdio")
