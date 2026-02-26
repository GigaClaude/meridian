"""StorageLayer — Qdrant (vector) + SQLite (graph/relational) + Episodic (JSON on disk).

This is the data backbone. No LLM calls here — just CRUD and search.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    PointVectors,
    VectorParams,
)

from .config import config
from .schemas import (
    Checkpoint,
    Decision,
    Entity,
    EpisodicSession,
    MemoryRecord,
    Relation,
    Warning,
    gen_id,
)

logger = logging.getLogger("meridian.storage")

# Patterns that indicate ephemeral/volatile content
import re

_VOLATILE_PATTERNS = [
    re.compile(r'\bport\s+\d{4,5}\b', re.IGNORECASE),        # "port 18112"
    re.compile(r'\blocalhost:\d{4,5}\b'),                      # "localhost:6333"
    re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+'),  # "192.168.1.1:8080"
    re.compile(r'\bPID\s*[:=]?\s*\d+\b', re.IGNORECASE),       # "PID: 12345" or "PID=12345"
    re.compile(r'(?:^|\s)/tmp/\S+'),                            # "/tmp/anything"
    re.compile(r'(?:^|\s)/var/run/\S+'),                        # "/var/run/anything"
    re.compile(r'\brunning\s+(?:on|at)\s+\S+:\d+\b', re.IGNORECASE),  # "running on host:port"
    re.compile(r'\blistening\s+on\s+\S+:\d+\b', re.IGNORECASE),       # "listening on :8080"
]

# Content types that are almost always stable (never volatile)
_STABLE_TYPES = {"decision", "pattern", "warning"}

# ── Quality gate: noise detection ──

_SPEAKER_PREFIX = re.compile(r'^\[.+?\]:')
_MIGRATED_PREFIX = re.compile(r'^\[migrated', re.IGNORECASE)
_CODE_INDICATORS = ['```', 'def ', 'import ', 'if __name__', 'class ', 'async def ', 'return ']
_FILLER_WORDS = ['lol', 'hmm', 'okay so', 'yeah', 'let me', 'oh wait', 'haha', 'nah']
_CONVERSATIONAL_PATTERNS = [
    re.compile(r'\b(you|your)\b.*\?', re.IGNORECASE),
    re.compile(r'^(ok|okay|sure|yeah|yep|nope|nah)\b', re.IGNORECASE),
    re.compile(r'let me (check|look|think|try)', re.IGNORECASE),
]


def _is_noisy(content: str) -> tuple[bool, list[str]]:
    """Quality gate: detect raw transcript noise that shouldn't be stored as memory.

    Returns (is_noisy, [reasons]).

    Tiered approach:
    - Instant NOISE: speaker prefixes ([Giga]:), multi-turn brackets
    - Allowlisted: [migrated...] prefix, code-containing content
    - 2+ signals required for: filler words, length, conversational patterns
    """
    # Allowlists
    if _MIGRATED_PREFIX.match(content):
        return False, []
    if any(indicator in content for indicator in _CODE_INDICATORS):
        if not _SPEAKER_PREFIX.match(content):
            return False, []

    # Instant noise: speaker prefix
    if _SPEAKER_PREFIX.match(content):
        return True, ["speaker_prefix"]

    # Instant noise: multi-turn
    bracket_speakers = re.findall(r'\[.+?\]:', content)
    if len(bracket_speakers) > 1:
        return True, ["multi_turn_speakers"]

    # Accumulating signals (need 2+)
    signals = []
    word_count = len(content.split())

    if word_count < 5:
        signals.append("too_short")
    if word_count > 500:
        signals.append("too_long")

    filler_count = sum(1 for f in _FILLER_WORDS if f in content.lower())
    if filler_count >= 2:
        signals.append("filler_words")

    for pattern in _CONVERSATIONAL_PATTERNS:
        if pattern.search(content):
            signals.append("conversational")
            break

    if content.strip().endswith('?') and '\n' not in content.strip() and word_count < 30:
        signals.append("standalone_question")

    return len(signals) >= 2, signals


def _detect_volatile(content: str) -> bool:
    """Detect if memory content contains ephemeral facts.

    Returns True if the content references ports, PIDs, temp paths, or
    runtime-specific details that are likely to change between sessions.
    Decisions, patterns, and warnings are never auto-marked volatile.
    """
    matches = sum(1 for p in _VOLATILE_PATTERNS if p.search(content))
    return matches >= 2  # Need 2+ signals to avoid false positives on casual port mentions


class StorageLayer:
    """Unified access to all three storage tiers."""

    def __init__(self, data_dir: Path | None = None, project_id: str | None = None):
        self.data_dir = data_dir or config.data_dir
        self.project_id = project_id or config.project_id
        self.db_path = self.data_dir / "meridian.db"
        self.episodes_dir = self.data_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        # Ollama URL for embeddings (set before _ensure_collection which may probe dim)
        self._ollama_url = config.ollama_url
        self._embed_model = config.embed_model

        # Qdrant client (sync — runs fast, no need for async)
        self.qdrant = QdrantClient(url=config.qdrant_url, timeout=30, check_compatibility=False)
        self._ensure_collection()
        self._http: httpx.AsyncClient | None = None

        # SQLite connection (created async)
        self._db: aiosqlite.Connection | None = None

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if config.qdrant_collection not in collections:
            # Probe embed model for actual vector dimension
            dim = self._probe_embed_dim()
            self.qdrant.create_collection(
                collection_name=config.qdrant_collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {config.qdrant_collection} (dim={dim})")

    def _probe_embed_dim(self) -> int:
        """Get embedding dimension from Ollama model. Falls back to 1024 (mxbai-embed-large)."""
        try:
            import httpx as _httpx
            resp = _httpx.post(
                f"{self._ollama_url}/api/embed",
                json={"model": self._embed_model, "input": "dimension probe"},
                timeout=30.0,
            )
            resp.raise_for_status()
            return len(resp.json()["embeddings"][0])
        except Exception as e:
            logger.warning(f"Could not probe embed dimension: {e}. Defaulting to 1024.")
            return 1024

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=60.0)
        return self._http

    async def init_db(self):
        """Initialize SQLite database and create tables."""
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        # Migrations — additive only, safe to re-run
        await self._migrate_add_checkpoint_source()
        logger.info(f"SQLite initialized at {self.db_path}")

    async def _migrate_add_checkpoint_source(self):
        """Add source column to checkpoints table if missing (multi-agent support)."""
        try:
            async with self._db.execute("PRAGMA table_info(checkpoints)") as cursor:
                columns = {row["name"] for row in await cursor.fetchall()}
            if "source" not in columns:
                await self._db.execute("ALTER TABLE checkpoints ADD COLUMN source TEXT")
                await self._db.commit()
                logger.info("Migration: added source column to checkpoints table")
        except Exception as e:
            logger.warning(f"Checkpoint source migration skipped: {e}")

    async def close(self):
        if self._db:
            await self._db.close()
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ── Embedding ──

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector via Ollama. Falls back to zero vector if Ollama is down."""
        try:
            http = await self._get_http()
            resp = await http.post(
                f"{self._ollama_url}/api/embed",
                json={"model": self._embed_model, "input": text, "keep_alive": "10m"},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"][0]
        except Exception as e:
            logger.warning(f"Embedding failed (Ollama down?): {e}")
            # Return zero vector — memory will be stored but not searchable until re-embedded
            return [0.0] * 1024

    # ── Vector Store (Qdrant) ──

    async def store_memory(self, record: MemoryRecord) -> str:
        """Store a memory record with its embedding in Qdrant."""
        vector = await self.embed(record.content)
        # Qdrant requires UUID or int point IDs
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, record.id))
        point = PointStruct(
            id=point_uuid,
            vector=vector,
            payload={
                "mem_id": record.id,
                "content": record.content,
                "type": record.type,
                "tags": record.tags,
                "project_id": record.project_id,
                "importance": record.importance,
                "source": record.source,
                "volatile": record.volatile,
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
                "superseded_by": record.superseded_by,
                "session_id": record.session_id,
                "related_ids": record.related_ids,
            },
        )
        self.qdrant.upsert(
            collection_name=config.qdrant_collection,
            points=[point],
        )
        logger.info(f"Stored memory {record.id}: {record.type}")
        return record.id

    async def get_all_tags(self, project_id: str | None = None) -> set[str]:
        """Return the set of all tags in the collection for a project."""
        pid = project_id or self.project_id
        pfilter = Filter(must=[
            FieldCondition(key="project_id", match=MatchValue(value=pid)),
        ])
        tags: set[str] = set()
        offset = None
        while True:
            result = self.qdrant.scroll(
                collection_name=config.qdrant_collection,
                scroll_filter=pfilter,
                limit=250,
                offset=offset,
                with_payload=["tags"],
            )
            points, next_offset = result
            for p in points:
                for t in p.payload.get("tags", []):
                    tags.add(t)
            if next_offset is None:
                break
            offset = next_offset
        return tags

    def _extract_tag_matches(self, query: str, known_tags: set[str]) -> list[str]:
        """Find tags that appear in the query (case-insensitive, whole-token).

        For multi-part tags (e.g. 'claude-walker'), also matches camelCase
        smashed forms ('ClaudeWalker'). Short tags (<5 chars) require exact
        token match to avoid false positives ('stt' in 'tts', etc).
        """
        query_lower = query.lower()
        # Normalize query: replace common separators with spaces for tokenization
        normalized = query_lower.replace("-", " ").replace("_", " ").replace("/", " ")
        query_tokens = set(normalized.split())

        matched: list[str] = []
        for tag in known_tags:
            tag_lower = tag.lower()
            tag_bare = tag_lower.replace("-", "").replace("_", "")
            is_compound = "-" in tag or "_" in tag  # multi-part tag

            if is_compound:
                # Compound tags: match substring or camelCase smash
                if tag_lower in query_lower:
                    matched.append(tag)
                elif tag_bare in query_lower.replace(" ", "").replace("-", "").replace("_", ""):
                    matched.append(tag)
            elif len(tag) >= 5:
                # Longer simple tags: safe to do substring match
                if tag_lower in query_tokens:
                    matched.append(tag)
            else:
                # Short simple tags (api, tts, stt, git, gpu...): exact token only
                if tag_lower in query_tokens:
                    matched.append(tag)
        return matched

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        scope: str = "all",
        project_id: str | None = None,
        tag_filter: list[str] | None = None,
        source_filter: str | None = None,
        exclude_sources: list[str] | None = None,
    ) -> list[dict]:
        """Semantic search across memories, optionally filtered by tags and/or source agent.

        Args:
            exclude_sources: List of source values to exclude from results.
                Used by recall() to filter out episodic_ingest noise.
        """
        vector = await self.embed(query)
        filters = []
        must_not = []
        pid = project_id or self.project_id
        filters.append(FieldCondition(key="project_id", match=MatchValue(value=pid)))
        if scope != "all":
            # Map scope names to memory types
            scope_map = {
                "decisions": "decision",
                "patterns": "pattern",
                "debug": "debug",
                "entities": "entity",
                "warnings": "warning",
            }
            if scope in scope_map:
                filters.append(
                    FieldCondition(key="type", match=MatchValue(value=scope_map[scope]))
                )
        if tag_filter:
            # Require at least one of the matched tags
            filters.append(
                FieldCondition(key="tags", match=MatchAny(any=tag_filter))
            )
        if source_filter:
            filters.append(
                FieldCondition(key="source", match=MatchValue(value=source_filter))
            )
        if exclude_sources:
            for src in exclude_sources:
                must_not.append(
                    FieldCondition(key="source", match=MatchValue(value=src))
                )

        if filters or must_not:
            search_filter = Filter(
                must=filters if filters else None,
                must_not=must_not if must_not else None,
            )
        else:
            search_filter = None
        results = self.qdrant.query_points(
            collection_name=config.qdrant_collection,
            query=vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "id": r.payload.get("mem_id", str(r.id)),
                "score": r.score,
                "content": r.payload.get("content", ""),
                "type": r.payload.get("type", ""),
                "tags": r.payload.get("tags", []),
                "importance": r.payload.get("importance", 3),
                "source": r.payload.get("source"),
                "volatile": r.payload.get("volatile", False),
                "created_at": r.payload.get("created_at", ""),
            }
            for r in results.points
            if not r.payload.get("superseded_by")  # Skip superseded memories
        ]

    def _mem_id_to_uuid(self, memory_id: str) -> str:
        """Convert our string memory ID to a Qdrant-compatible UUID."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, memory_id))

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from Qdrant."""
        point_uuid = self._mem_id_to_uuid(memory_id)
        self.qdrant.delete(
            collection_name=config.qdrant_collection,
            points_selector=[point_uuid],
        )
        return True

    async def update_memory_payload(self, memory_id: str, updates: dict) -> bool:
        """Update payload fields on an existing memory point."""
        point_uuid = self._mem_id_to_uuid(memory_id)
        self.qdrant.set_payload(
            collection_name=config.qdrant_collection,
            payload=updates,
            points=[point_uuid],
        )
        return True

    async def get_memory_count(self, project_id: str | None = None) -> int:
        """Get total memory count for a project."""
        info = self.qdrant.get_collection(config.qdrant_collection)
        return info.points_count or 0

    # ── SQLite: Knowledge Graph ──

    async def store_entity(self, entity: Entity) -> str:
        await self._db.execute(
            "INSERT OR REPLACE INTO entities (id, name, type, metadata, project_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (entity.id, entity.name, entity.type, json.dumps(entity.metadata),
             entity.project_id, entity.created_at.isoformat(), entity.updated_at.isoformat()),
        )
        await self._db.commit()
        return entity.id

    async def store_relation(self, relation: Relation) -> str:
        await self._db.execute(
            "INSERT OR REPLACE INTO relations (id, source_id, target_id, type, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (relation.id, relation.source_id, relation.target_id, relation.type,
             json.dumps(relation.metadata), relation.created_at.isoformat()),
        )
        await self._db.commit()
        return relation.id

    async def store_decision(self, decision: Decision) -> str:
        await self._db.execute(
            "INSERT OR REPLACE INTO decisions (id, title, rationale, alternatives, status, superseded_by, tags, importance, project_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (decision.id, decision.title, decision.rationale, json.dumps(decision.alternatives),
             decision.status, decision.superseded_by, json.dumps(decision.tags),
             decision.importance, decision.project_id, decision.created_at.isoformat()),
        )
        await self._db.commit()
        return decision.id

    async def store_warning(self, warning: Warning) -> str:
        await self._db.execute(
            "INSERT OR REPLACE INTO warnings (id, content, severity, related_entity, resolved, project_id, created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (warning.id, warning.content, warning.severity, warning.related_entity,
             warning.resolved, warning.project_id, warning.created_at.isoformat(),
             warning.expires_at.isoformat() if warning.expires_at else None),
        )
        await self._db.commit()
        return warning.id

    async def store_checkpoint(self, checkpoint: Checkpoint) -> str:
        await self._db.execute(
            "INSERT INTO checkpoints (id, session_id, project_id, source, task_state, decisions, warnings, next_steps, working_set, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (checkpoint.id, checkpoint.session_id, checkpoint.project_id,
             checkpoint.source, checkpoint.task_state,
             json.dumps(checkpoint.decisions), json.dumps(checkpoint.warnings),
             json.dumps(checkpoint.next_steps), json.dumps(checkpoint.working_set),
             checkpoint.created_at.isoformat()),
        )
        await self._db.commit()
        logger.info(f"Stored checkpoint {checkpoint.id} (source={checkpoint.source})")
        return checkpoint.id

    async def get_latest_checkpoint(self, project_id: str | None = None, source: str | None = None) -> dict | None:
        """Get latest checkpoint, optionally scoped to an agent.

        If source is provided, returns that agent's latest checkpoint.
        If source is None, returns any agent's latest checkpoint (backward-compatible).
        """
        pid = project_id or self.project_id
        if source:
            query = "SELECT * FROM checkpoints WHERE project_id = ? AND source = ? ORDER BY created_at DESC LIMIT 1"
            params = (pid, source)
        else:
            query = "SELECT * FROM checkpoints WHERE project_id = ? ORDER BY created_at DESC LIMIT 1"
            params = (pid,)
        async with self._db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "id": row["id"],
                "session_id": row["session_id"],
                "source": row["source"] if "source" in row.keys() else None,
                "task_state": row["task_state"],
                "decisions": json.loads(row["decisions"]),
                "warnings": json.loads(row["warnings"]),
                "next_steps": json.loads(row["next_steps"]),
                "working_set": json.loads(row["working_set"]),
                "created_at": row["created_at"],
            }

    async def get_recent_decisions(self, project_id: str | None = None, limit: int = 5) -> list[dict]:
        pid = project_id or self.project_id
        async with self._db.execute(
            "SELECT * FROM decisions WHERE project_id = ? AND status = 'active' ORDER BY created_at DESC LIMIT ?",
            (pid, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "rationale": row["rationale"],
                    "tags": json.loads(row["tags"]),
                    "importance": row["importance"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    async def get_active_warnings(self, project_id: str | None = None) -> list[dict]:
        pid = project_id or self.project_id
        async with self._db.execute(
            "SELECT * FROM warnings WHERE project_id = ? AND resolved = 0 ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, created_at DESC",
            (pid,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "severity": row["severity"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    async def query_graph(self, entity_name: str, relation_type: str | None = None, depth: int = 2) -> dict:
        """Traverse the entity relationship graph from a starting entity."""
        entities_found = []
        relations_found = []
        visited = set()

        async def _traverse(name: str, current_depth: int):
            if current_depth > depth or name in visited:
                return
            visited.add(name)

            # Find entity
            async with self._db.execute(
                "SELECT * FROM entities WHERE name = ? AND project_id = ?",
                (name, self.project_id),
            ) as cursor:
                entity_row = await cursor.fetchone()
                if not entity_row:
                    return
                entities_found.append({
                    "id": entity_row["id"],
                    "name": entity_row["name"],
                    "type": entity_row["type"],
                    "metadata": json.loads(entity_row["metadata"]) if entity_row["metadata"] else {},
                })

            # Find relations
            rel_query = "SELECT r.*, e.name as target_name FROM relations r JOIN entities e ON r.target_id = e.id WHERE r.source_id = ?"
            params = [entity_row["id"]]
            if relation_type:
                rel_query += " AND r.type = ?"
                params.append(relation_type)

            async with self._db.execute(rel_query, params) as cursor:
                rels = await cursor.fetchall()
                for rel in rels:
                    relations_found.append({
                        "source": name,
                        "target": rel["target_name"],
                        "type": rel["type"],
                    })
                    await _traverse(rel["target_name"], current_depth + 1)

            # Also check reverse relations
            rev_query = "SELECT r.*, e.name as source_name FROM relations r JOIN entities e ON r.source_id = e.id WHERE r.target_id = ?"
            params = [entity_row["id"]]
            if relation_type:
                rev_query += " AND r.type = ?"
                params.append(relation_type)

            async with self._db.execute(rev_query, params) as cursor:
                rels = await cursor.fetchall()
                for rel in rels:
                    relations_found.append({
                        "source": rel["source_name"],
                        "target": name,
                        "type": rel["type"],
                    })
                    await _traverse(rel["source_name"], current_depth + 1)

        await _traverse(entity_name, 0)
        return {"entities": entities_found, "relations": relations_found}

    async def find_entity_by_name(self, name: str) -> dict | None:
        async with self._db.execute(
            "SELECT * FROM entities WHERE name = ? AND project_id = ?",
            (name, self.project_id),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return {"id": row["id"], "name": row["name"], "type": row["type"]}

    async def find_entities_in_text(self, text: str, min_name_length: int = 3) -> list[dict]:
        """Find known entities whose names appear in the given text.

        Returns entities sorted by name length descending (longer matches first).
        Only matches entities with names >= min_name_length to avoid noise.
        """
        text_lower = text.lower()
        async with self._db.execute(
            "SELECT id, name, type FROM entities WHERE project_id = ? AND LENGTH(name) >= ?",
            (self.project_id, min_name_length),
        ) as cursor:
            rows = await cursor.fetchall()

        matches = []
        for row in rows:
            if row["name"].lower() in text_lower:
                matches.append({
                    "id": row["id"],
                    "name": row["name"],
                    "type": row["type"],
                })
        # Longer names first (more specific matches)
        matches.sort(key=lambda e: len(e["name"]), reverse=True)
        return matches

    async def get_related_entity_names(self, entity_name: str, depth: int = 1) -> set[str]:
        """Get names of entities related to the given entity (shallow traversal).

        Returns a set of entity names connected within `depth` hops.
        """
        result = await self.query_graph(entity_name, depth=depth)
        names = set()
        for ent in result.get("entities", []):
            names.add(ent["name"])
        for rel in result.get("relations", []):
            names.add(rel["source"])
            names.add(rel["target"])
        names.discard(entity_name)  # Don't include the starting entity
        return names

    # ── Episodic Store ──

    async def store_episode(self, session: EpisodicSession, transcript: list[dict]) -> str:
        """Store a session transcript as a compressed JSON file."""
        date_dir = self.episodes_dir / session.started_at.strftime("%Y-%m-%d")
        date_dir.mkdir(exist_ok=True)

        episode_data = {
            "session_id": session.session_id,
            "project_id": session.project_id,
            "started_at": session.started_at.isoformat(),
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "summary": session.summary,
            "key_events": session.key_events,
            "transcript_hash": session.transcript_hash,
            "transcript": transcript,
        }

        filepath = date_dir / f"{session.session_id}.json.gz"
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(episode_data, f)

        # Update day index
        await self._update_episode_index(date_dir, session)
        logger.info(f"Stored episode {session.session_id}")
        return session.session_id

    async def _update_episode_index(self, date_dir: Path, session: EpisodicSession):
        """Update the daily episode index."""
        index_path = date_dir / "index.json"
        index = []
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)

        index.append({
            "session_id": session.session_id,
            "project_id": session.project_id,
            "started_at": session.started_at.isoformat(),
            "summary": session.summary,
            "key_events_count": len(session.key_events),
        })

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    async def search_episodes(self, query: str, limit: int = 5) -> list[dict]:
        """Search episode indexes for matching sessions (keyword-based)."""
        results = []
        query_lower = query.lower()

        for date_dir in sorted(self.episodes_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            index_path = date_dir / "index.json"
            if not index_path.exists():
                continue
            with open(index_path) as f:
                index = json.load(f)
            for entry in index:
                if query_lower in entry.get("summary", "").lower():
                    results.append(entry)
                    if len(results) >= limit:
                        return results
        return results

    async def load_episode(self, session_id: str) -> dict | None:
        """Load a full episode by session ID."""
        for date_dir in self.episodes_dir.iterdir():
            if not date_dir.is_dir():
                continue
            filepath = date_dir / f"{session_id}.json.gz"
            if filepath.exists():
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    return json.load(f)
        return None

    # ── Project Management ──

    async def list_projects(self) -> list[dict]:
        """List projects that have memories stored."""
        async with self._db.execute(
            "SELECT DISTINCT project_id FROM checkpoints UNION SELECT DISTINCT project_id FROM decisions UNION SELECT DISTINCT project_id FROM entities",
        ) as cursor:
            rows = await cursor.fetchall()
            return [{"project_id": row["project_id"]} for row in rows]

    async def project_status(self, project_id: str) -> dict:
        """Get memory stats for a project."""
        checkpoint = await self.get_latest_checkpoint(project_id)
        warnings = await self.get_active_warnings(project_id)
        memory_count = await self.get_memory_count(project_id)

        return {
            "project_id": project_id,
            "memory_count": memory_count,
            "active_warnings": len(warnings),
            "last_checkpoint": checkpoint["created_at"] if checkpoint else None,
            "last_task": checkpoint["task_state"] if checkpoint else None,
        }

    # ── Cold Storage (warm→cold demotion + promotion) ──

    async def demote_to_cold(self, memory_id: str, reason: str = "aged_out") -> bool:
        """Move a memory from warm (Qdrant) to cold (SQLite archive).

        The memory is removed from Qdrant vector search but preserved in
        SQLite for keyword-based cold recall. Nothing is lost.
        """
        # First, retrieve the full payload from Qdrant
        point_uuid = self._mem_id_to_uuid(memory_id)
        try:
            points = self.qdrant.retrieve(
                collection_name=config.qdrant_collection,
                ids=[point_uuid],
                with_payload=True,
            )
        except Exception as e:
            logger.warning(f"Could not retrieve {memory_id} for demotion: {e}")
            return False

        if not points:
            logger.warning(f"Memory {memory_id} not found in Qdrant for demotion")
            return False

        payload = points[0].payload

        # Archive to SQLite cold_memories table
        await self._db.execute(
            """INSERT OR REPLACE INTO cold_memories
               (id, mem_id, content, type, tags, importance, source, volatile,
                project_id, created_at, demoted_at, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                payload.get("mem_id", memory_id),
                payload.get("mem_id", memory_id),
                payload.get("content", ""),
                payload.get("type", "note"),
                json.dumps(payload.get("tags", [])),
                payload.get("importance", 2),
                payload.get("source"),
                1 if payload.get("volatile") else 0,
                payload.get("project_id", self.project_id),
                payload.get("created_at", ""),
                datetime.now(timezone.utc).isoformat(),
                reason,
            ),
        )
        await self._db.commit()

        # Remove from Qdrant
        self.qdrant.delete(
            collection_name=config.qdrant_collection,
            points_selector=[point_uuid],
        )
        logger.info(f"Demoted {memory_id} to cold storage: {reason}")
        return True

    async def promote_from_cold(self, memory_id: str) -> bool:
        """Promote a memory from cold back to warm (Qdrant).

        Bumps importance by 1 (capped at 5) — the fact it was needed after
        demotion means the original importance was too low.
        """
        async with self._db.execute(
            "SELECT * FROM cold_memories WHERE mem_id = ?", (memory_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return False

        # Bump importance (self-correcting: recalled cold memory was undervalued)
        new_importance = min(5, row["importance"] + 1)

        record = MemoryRecord(
            id=row["mem_id"],
            content=row["content"],
            type=row["type"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            importance=new_importance,
            source=row["source"],
            volatile=bool(row["volatile"]),
            project_id=row["project_id"],
        )
        if row["created_at"]:
            try:
                record.created_at = datetime.fromisoformat(row["created_at"])
            except (ValueError, TypeError):
                pass

        await self.store_memory(record)

        # Remove from cold storage
        await self._db.execute("DELETE FROM cold_memories WHERE mem_id = ?", (memory_id,))
        await self._db.commit()
        logger.info(f"Promoted {memory_id} from cold to warm (importance {row['importance']}→{new_importance})")
        return True

    async def search_cold(self, query: str, limit: int = 5) -> list[dict]:
        """Keyword search across cold storage memories.

        Used as fallback when warm recall returns insufficient results.
        """
        query_terms = query.lower().split()
        # Build LIKE conditions for each query term
        conditions = " AND ".join(
            f"(LOWER(content) LIKE ? OR LOWER(tags) LIKE ?)"
            for _ in query_terms
        )
        params = []
        for term in query_terms:
            like_term = f"%{term}%"
            params.extend([like_term, like_term])

        sql = f"""SELECT * FROM cold_memories
                  WHERE project_id = ? AND {conditions}
                  ORDER BY importance DESC, created_at DESC
                  LIMIT ?"""

        results = []
        async with self._db.execute(sql, [self.project_id] + params + [limit]) as cursor:
            async for row in cursor:
                results.append({
                    "id": row["mem_id"],
                    "content": row["content"],
                    "type": row["type"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "importance": row["importance"],
                    "source": row["source"],
                    "volatile": bool(row["volatile"]),
                    "created_at": row["created_at"],
                    "demoted_at": row["demoted_at"],
                    "cold": True,
                })

        # Track access for any returned results
        if results:
            now = datetime.now(timezone.utc).isoformat()
            for r in results:
                await self._db.execute(
                    "UPDATE cold_memories SET access_count = access_count + 1, last_accessed = ? WHERE mem_id = ?",
                    (now, r["id"]),
                )
            await self._db.commit()

        return results

    async def sweep_demotions(self, pinned_ids: set[str] | None = None) -> dict:
        """Run a demotion sweep across warm storage.

        Demotion criteria:
        1. Volatile + older than 7 days + importance ≤ 3
        2. Non-volatile + older than 30 days + importance ≤ 2
        3. Superseded by a newer memory

        Pinned IDs (from active working_set) are exempt from demotion.
        Returns counts of demoted memories.
        """
        pinned = pinned_ids or set()
        now = datetime.now(timezone.utc)
        demoted = {"volatile": 0, "low_importance": 0, "superseded": 0}

        # Scroll through all memories in Qdrant
        offset = None
        candidates = []
        pfilter = Filter(must=[
            FieldCondition(key="project_id", match=MatchValue(value=self.project_id)),
        ])

        while True:
            result = self.qdrant.scroll(
                collection_name=config.qdrant_collection,
                scroll_filter=pfilter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            points, next_offset = result
            for p in points:
                mem_id = p.payload.get("mem_id", "")
                if mem_id in pinned:
                    continue

                importance = p.payload.get("importance", 3)
                is_volatile = p.payload.get("volatile", False)
                is_superseded = bool(p.payload.get("superseded_by"))
                created_at = p.payload.get("created_at", "")

                days_old = 0
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at)
                        days_old = (now - dt).total_seconds() / 86400
                    except (ValueError, TypeError):
                        pass

                reason = None
                category = None

                # Superseded memories — always demote
                if is_superseded:
                    reason = "superseded"
                    category = "superseded"
                # Volatile + old + low importance
                elif is_volatile and days_old > 7 and importance <= 3:
                    reason = f"volatile_aged_out (age={int(days_old)}d, imp={importance})"
                    category = "volatile"
                # Non-volatile + very old + low importance
                elif not is_volatile and days_old > 30 and importance <= 2:
                    reason = f"low_importance_aged_out (age={int(days_old)}d, imp={importance})"
                    category = "low_importance"

                if reason:
                    candidates.append((mem_id, reason, category))

            if next_offset is None:
                break
            offset = next_offset

        # Execute demotions
        for mem_id, reason, category in candidates:
            success = await self.demote_to_cold(mem_id, reason)
            if success:
                demoted[category] += 1

        total = sum(demoted.values())
        if total > 0:
            logger.info(f"Demotion sweep: {total} memories moved to cold ({demoted})")
        return demoted

    # ── High-level Operations (called by MCP tools) ──

    async def remember(self, args: dict, workers=None) -> dict:
        """Store a new memory. Called by memory_remember MCP tool.

        Includes quality gate, conflict detection, and volatile tagging.
        Rejects raw transcript noise before it hits the vector store.
        """
        content = args["content"]

        # Quality gate: reject raw transcript noise
        is_noisy, noise_reasons = _is_noisy(content)
        if is_noisy:
            logger.warning(f"Quality gate REJECTED memory: {noise_reasons} — {content[:80]}")
            return {
                "id": None,
                "stored": False,
                "rejected": True,
                "reasons": noise_reasons,
            }

        record = MemoryRecord(
            content=content,
            type=args["type"],
            tags=args.get("tags", []),
            importance=args.get("importance", 3),
            source=args.get("source"),
            related_ids=args.get("related_to", []),
            project_id=self.project_id,
        )

        # Auto-detect volatile content (ephemeral facts that decay faster)
        # Skip for stable types — decisions, patterns, and warnings are never auto-volatile
        if not record.volatile and record.type not in _STABLE_TYPES:
            record.volatile = _detect_volatile(record.content)
            if record.volatile and "volatile" not in record.tags:
                record.tags = list(record.tags) + ["volatile"]

        # Conflict detection: check for highly similar existing memories
        superseded = []
        try:
            similar = await self.search_memories(
                record.content, limit=3, scope="all",
            )
            for s in similar:
                if s["score"] > 0.92 and s["id"] != record.id:
                    # High similarity — supersede the old memory
                    old_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, s["id"]))
                    try:
                        self.qdrant.set_payload(
                            collection_name=config.qdrant_collection,
                            payload={"superseded_by": record.id},
                            points=[old_uuid],
                        )
                        superseded.append(s["id"])
                        logger.info(f"Superseded {s['id']} (score={s['score']:.3f}) by new memory")
                    except Exception as e:
                        logger.warning(f"Failed to supersede {s['id']}: {e}")
        except Exception as e:
            logger.warning(f"Conflict detection failed (non-fatal): {e}")

        mem_id = await self.store_memory(record)

        # If it's a decision, also store in decisions table
        if record.type == "decision":
            decision = Decision(
                id=mem_id,
                title=record.content[:100],
                rationale=record.content,
                tags=record.tags,
                importance=record.importance,
                project_id=self.project_id,
            )
            await self.store_decision(decision)

        # If it's a warning, also store in warnings table
        if record.type == "warning":
            warning = Warning(
                content=record.content,
                severity="high" if record.importance >= 4 else "medium",
                project_id=self.project_id,
            )
            await self.store_warning(warning)

        # Kick off async entity extraction if workers available
        if workers:
            asyncio.create_task(workers.extract_and_store_entities(record.content, self))

        result = {"id": mem_id, "stored": True}
        if superseded:
            result["superseded"] = superseded
        return result

    async def checkpoint(self, args: dict) -> dict:
        """Save a checkpoint. Called by memory_checkpoint MCP tool."""
        cp = Checkpoint(
            session_id=args.get("session_id", gen_id("ses")),
            project_id=self.project_id,
            source=args.get("source"),
            task_state=args["task_state"],
            decisions=args.get("decisions", []),
            warnings=args.get("warnings", []),
            next_steps=args.get("next_steps", []),
            working_set=args.get("working_set", {}),
        )
        cp_id = await self.store_checkpoint(cp)
        return {"checkpoint_id": cp_id, "stored": True}

    async def forget(self, args: dict) -> dict:
        """Remove or deprecate a memory. Called by memory_forget MCP tool."""
        memory_id = args["id"]
        reason = args.get("reason", "")
        try:
            await self.delete_memory(memory_id)
            # Also try to remove from decisions/warnings tables
            await self._db.execute("DELETE FROM decisions WHERE id = ?", (memory_id,))
            await self._db.execute("DELETE FROM warnings WHERE id = ?", (memory_id,))
            await self._db.commit()
            logger.info(f"Forgot memory {memory_id}: {reason}")
            return {"forgotten": True}
        except Exception as e:
            logger.error(f"Failed to forget {memory_id}: {e}")
            return {"forgotten": False, "error": str(e)}

    async def update(self, args: dict) -> dict:
        """Update an existing memory. Called by memory_update MCP tool."""
        memory_id = args["id"]
        updates = args.get("updates", {})

        payload_updates = {}
        if "content" in updates:
            payload_updates["content"] = updates["content"]
        if "tags" in updates:
            payload_updates["tags"] = updates["tags"]
        if "importance" in updates:
            payload_updates["importance"] = updates["importance"]
        if "superseded_by" in updates:
            payload_updates["superseded_by"] = updates["superseded_by"]
        payload_updates["updated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            await self.update_memory_payload(memory_id, payload_updates)

            # If content changed, re-embed
            if "content" in updates:
                new_vector = await self.embed(updates["content"])
                point_uuid = self._mem_id_to_uuid(memory_id)
                self.qdrant.update_vectors(
                    collection_name=config.qdrant_collection,
                    points=[PointVectors(id=point_uuid, vector=new_vector)],
                )

            return {"updated": True}
        except Exception as e:
            logger.error(f"Failed to update {memory_id}: {e}")
            return {"updated": False, "error": str(e)}


# ── SQLite Schema ──

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    metadata TEXT,
    project_id TEXT DEFAULT 'default',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES entities(id),
    target_id TEXT REFERENCES entities(id),
    type TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    rationale TEXT NOT NULL,
    alternatives TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active',
    superseded_by TEXT REFERENCES decisions(id),
    tags TEXT DEFAULT '[]',
    importance INTEGER DEFAULT 3,
    project_id TEXT DEFAULT 'default',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS warnings (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    severity TEXT DEFAULT 'medium',
    related_entity TEXT REFERENCES entities(id),
    resolved BOOLEAN DEFAULT 0,
    project_id TEXT DEFAULT 'default',
    created_at TEXT DEFAULT (datetime('now')),
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    project_id TEXT DEFAULT 'default',
    source TEXT,
    task_state TEXT,
    decisions TEXT DEFAULT '[]',
    warnings TEXT DEFAULT '[]',
    next_steps TEXT DEFAULT '[]',
    working_set TEXT DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cold_memories (
    id TEXT PRIMARY KEY,
    mem_id TEXT NOT NULL,
    content TEXT NOT NULL,
    type TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    importance INTEGER DEFAULT 2,
    source TEXT,
    volatile BOOLEAN DEFAULT 0,
    project_id TEXT DEFAULT 'default',
    created_at TEXT,
    demoted_at TEXT DEFAULT (datetime('now')),
    reason TEXT,
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT
);

CREATE INDEX IF NOT EXISTS idx_entities_project ON entities(project_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project_id, status);
CREATE INDEX IF NOT EXISTS idx_warnings_project ON warnings(project_id, resolved);
CREATE INDEX IF NOT EXISTS idx_checkpoints_project ON checkpoints(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cold_memories_project ON cold_memories(project_id);
CREATE INDEX IF NOT EXISTS idx_cold_memories_content ON cold_memories(content);
"""
