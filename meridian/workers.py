"""Worker Pool — handles embedding, entity extraction, summarization, indexing.

Uses a local LLM via Ollama. High-throughput, lower-complexity tasks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import httpx

from .config import config
from .schemas import Entity, EpisodicSession, Relation, gen_id

if TYPE_CHECKING:
    from .storage import StorageLayer

logger = logging.getLogger("meridian.workers")


class WorkerPool:
    """Worker pool for extraction, summarization, and indexing."""

    def __init__(self, ollama_url: str | None = None, model: str | None = None):
        self.ollama_url = ollama_url or config.ollama_url
        self.model = model or config.worker_model
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=90.0)
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def _generate(self, prompt: str, system: str = "", max_tokens: int = 500) -> str:
        http = await self._get_http()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,
            },
        }
        try:
            resp = await http.post(f"{self.ollama_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.error(f"Worker generation failed: {e}")
            return ""

    async def extract_and_store_entities(self, content: str, storage: StorageLayer):
        """Extract entities and relationships from content and store them."""
        prompt = f"""Extract entities and relationships from this text. Return JSON only.

TEXT: {content}

Return format:
{{"entities": [{{"name": "...", "type": "service|file|concept|person|config"}}], "relations": [{{"source": "...", "target": "...", "type": "calls|depends_on|configured_by|implements|broke"}}]}}

JSON:"""

        system = "You are an entity extraction engine. Output ONLY valid JSON, no explanation."
        result = await self._generate(prompt, system=system, max_tokens=500)

        try:
            # Try to parse JSON from response
            # Handle cases where the model wraps in markdown code blocks
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()

            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            logger.warning(f"Failed to parse entity extraction result: {result[:200]}")
            return

        project_id = storage.project_id

        # Store entities
        entity_map = {}  # name -> id
        for ent in data.get("entities", []):
            name = ent.get("name", "").strip()
            etype = ent.get("type", "concept")
            if not name:
                continue

            # Check if entity already exists
            existing = await storage.find_entity_by_name(name)
            if existing:
                entity_map[name] = existing["id"]
                continue

            entity = Entity(name=name, type=etype, project_id=project_id)
            eid = await storage.store_entity(entity)
            entity_map[name] = eid

        # Store relations
        for rel in data.get("relations", []):
            source_name = rel.get("source", "").strip()
            target_name = rel.get("target", "").strip()
            rtype = rel.get("type", "related_to")

            source_id = entity_map.get(source_name)
            target_id = entity_map.get(target_name)

            if source_id and target_id:
                relation = Relation(
                    source_id=source_id,
                    target_id=target_id,
                    type=rtype,
                )
                await storage.store_relation(relation)

        logger.info(f"Extracted {len(entity_map)} entities, {len(data.get('relations', []))} relations")

    async def summarize_session(self, messages: list[dict]) -> dict:
        """Summarize a session transcript into key events and overall summary."""
        # Build a condensed version of the transcript
        transcript_text = ""
        for msg in messages[-50:]:  # Last 50 messages max
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."
            transcript_text += f"{role}: {content}\n"

        if not transcript_text.strip():
            return {"summary": "Empty session", "key_events": []}

        prompt = f"""Summarize this session transcript. Include:
1. A 1-2 sentence overall summary
2. Key events (decisions made, bugs found, features implemented)

TRANSCRIPT:
{transcript_text[:3000]}

Return JSON:
{{"summary": "...", "key_events": [{{"type": "decision|debug|feature|note", "content": "..."}}]}}

JSON:"""

        system = "You are a session summarizer. Output ONLY valid JSON."
        result = await self._generate(prompt, system=system, max_tokens=600)

        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            logger.warning(f"Failed to parse session summary: {result[:200]}")
            return {"summary": "Session summary unavailable", "key_events": []}

    async def classify_tags(self, content: str) -> list[str]:
        """Auto-tag a memory by domain/topic."""
        prompt = f"""Classify this text into 2-5 relevant tags from software development domains.

TEXT: {content}

Return JSON array of tag strings only:"""

        system = "Output ONLY a JSON array of strings. No explanation."
        result = await self._generate(prompt, system=system, max_tokens=100)

        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            tags = json.loads(cleaned)
            if isinstance(tags, list):
                return [str(t).lower().strip() for t in tags[:5]]
        except (json.JSONDecodeError, IndexError):
            pass
        return []

    async def process_messages(self, messages: list[dict], storage: StorageLayer):
        """Lightweight async processing of messages during a session."""
        # Extract entities from any substantial assistant messages
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        block.get("text", "") for block in content if isinstance(block, dict)
                    )
                if len(content) > 100:  # Only process substantial messages
                    await self.extract_and_store_entities(content, storage)

    async def process_session_transcript(self, messages: list[dict], storage: StorageLayer):
        """Full async post-session processing pipeline."""
        if not messages:
            return

        # 1. Summarize session
        summary_data = await self.summarize_session(messages)

        # 2. Store episode
        transcript_str = json.dumps(messages, default=str)
        transcript_hash = hashlib.sha256(transcript_str.encode()).hexdigest()

        session = EpisodicSession(
            project_id=storage.project_id,
            ended_at=datetime.utcnow(),
            summary=summary_data.get("summary", ""),
            key_events=summary_data.get("key_events", []),
            transcript_hash=f"sha256:{transcript_hash[:16]}",
        )

        # Convert messages to serializable format
        serializable_messages = []
        for msg in messages:
            serializable_messages.append({
                "role": msg.get("role", "unknown"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", datetime.utcnow().isoformat()),
            })

        await storage.store_episode(session, serializable_messages)

        # 3. Extract entities from full transcript
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            if len(content) > 200:
                await self.extract_and_store_entities(content, storage)

        # 4. Update global episode index
        await self._update_global_index(storage)

        logger.info(f"Post-session processing complete for {session.session_id}")

    async def _update_global_index(self, storage: StorageLayer):
        """Rebuild global episode index from daily indexes."""
        global_index = []
        episodes_dir = storage.episodes_dir

        for date_dir in sorted(episodes_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            index_path = date_dir / "index.json"
            if index_path.exists():
                with open(index_path) as f:
                    daily = json.load(f)
                    global_index.extend(daily)

        global_path = episodes_dir / "global_index.json"
        with open(global_path, "w") as f:
            json.dump(global_index, f, indent=2)

        logger.info(f"Updated global index: {len(global_index)} episodes")
