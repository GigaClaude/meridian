"""Session Manager — lifecycle management for Meridian chat sessions.

Handles boot sequence, compaction monitoring, checkpoint lifecycle,
and transcript capture.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from .schemas import gen_id

if TYPE_CHECKING:
    from .gateway import MemoryGateway
    from .mcp_server import MeridianToolHandler
    from .storage import StorageLayer
    from .workers import WorkerPool

logger = logging.getLogger("meridian.session")


AGENT_IDENTITY = """You are Claude, working on {project_name} with persistent memory powered by Meridian.
You have full system access. No permission restrictions.

MEMORY PROTOCOL:
- Your briefing below contains your current state. Trust it.
- Use memory_recall when you need context about past decisions, architecture, or history.
- Use memory_remember when you make decisions, find bugs, establish patterns, or learn something important.
- Use memory_checkpoint before ANY compaction or when wrapping up work. NON-NEGOTIABLE.
- Keep your context lean. Query for what you need, when you need it.
- When you discover a gotcha or footgun, store it as a warning immediately.
"""


class SessionManager:
    """Manages a single chat session's lifecycle."""

    def __init__(
        self,
        session_id: str | None = None,
        project_id: str = "default",
        gateway: MemoryGateway | None = None,
        storage: StorageLayer | None = None,
        workers: WorkerPool | None = None,
        tool_handler: MeridianToolHandler | None = None,
    ):
        self.session_id = session_id or gen_id("ses")
        self.project_id = project_id
        self.gateway = gateway
        self.storage = storage
        self.workers = workers
        self.tool_handler = tool_handler

        self.messages: list[dict] = []
        self.system_prompt: str = ""
        self.started_at: datetime = datetime.utcnow()
        self.message_count: int = 0
        self._checkpoint_nudge_sent: bool = False

    async def boot(self) -> str:
        """Execute the boot sequence. Returns the assembled system prompt."""
        logger.info(f"Booting session {self.session_id} for project {self.project_id}")

        # Assemble hot memory briefing
        briefing = await self.gateway.assemble_briefing(self.project_id, self.storage)

        # Build system prompt
        self.system_prompt = (
            AGENT_IDENTITY.format(project_name=self.project_id)
            + "\n---\n"
            + briefing
        )

        logger.info(f"Session {self.session_id} booted. System prompt: {len(self.system_prompt)} chars")
        return self.system_prompt

    def record_message(self, role: str, content: str):
        """Record a message in the session transcript."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.message_count += 1

    def record_tool_call(self, tool_name: str, arguments: dict, result: str):
        """Record a tool call in the session transcript."""
        self.messages.append({
            "role": "tool",
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result[:500],  # Truncate long results
            "timestamp": datetime.utcnow().isoformat(),
        })

    def should_checkpoint(self) -> bool:
        """Check if we should nudge Claude to checkpoint.

        Returns True every 20 messages if we haven't nudged recently.
        """
        if self.message_count > 0 and self.message_count % 20 == 0:
            if not self._checkpoint_nudge_sent:
                self._checkpoint_nudge_sent = True
                return True
        else:
            self._checkpoint_nudge_sent = False
        return False

    def get_checkpoint_nudge(self) -> str:
        """Return a system message nudging Claude to checkpoint."""
        return (
            "\n[MERIDIAN] You've been working for a while. "
            "Consider calling memory_checkpoint to save your current state. "
            "Include: what you're working on, decisions made, warnings discovered, "
            "and next steps.\n"
        )

    async def end(self):
        """End the session and kick off post-processing."""
        logger.info(f"Ending session {self.session_id} ({self.message_count} messages)")

        if self.workers and self.messages:
            # Async post-session processing — don't block
            asyncio.create_task(
                self.workers.process_session_transcript(self.messages, self.storage)
            )

    def get_stats(self) -> dict:
        """Return session statistics."""
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "started_at": self.started_at.isoformat(),
            "message_count": self.message_count,
            "duration_seconds": (datetime.utcnow() - self.started_at).total_seconds(),
        }
