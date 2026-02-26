"""Memory Gateway — synthesis layer.

Routes queries, synthesizes results, assembles hot memory briefings.
Uses a local LLM via Ollama for synthesis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx
import yaml

from .config import config

if TYPE_CHECKING:
    from .storage import StorageLayer

logger = logging.getLogger("meridian.gateway")

GATEWAY_SYSTEM_PROMPT = """Memory retrieval filter. Extract ONLY facts answering the query. Drop noise (expect 50-70% irrelevant).
Rules: cite everything [mem_id]. No citation=don't say it. Preserve specifics (paths,ports,commands,rationale). Newer overrides older on conflict. Weight imp4-5 heavily. Shorter=better. Prose, no bullets. Stop when answered."""


# JSON schema for structured briefing output via Ollama's format parameter.
# This forces the LLM to emit valid JSON matching this exact structure.
BRIEFING_SCHEMA = {
    "type": "object",
    "required": ["project", "latest_checkpoint", "recent_decisions", "active_warnings", "meridian_gotchas"],
    "properties": {
        "project": {"type": "string"},
        "latest_checkpoint": {
            "type": "object",
            "required": ["task", "decisions", "warnings", "next_steps", "working_set"],
            "properties": {
                "task": {"type": "string"},
                "decisions": {"type": "array", "items": {"type": "string"}},
                "warnings": {"type": "array", "items": {"type": "string"}},
                "next_steps": {"type": "array", "items": {"type": "string"}},
                "working_set": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string"}},
                        "endpoints": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
        "recent_decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "description", "rationale"],
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
        "active_warnings": {"type": "array", "items": {"type": "string"}},
        "port_assignments": {
            "type": "array",
            "items": {"type": "string"},
        },
        "migrated_memories_warning": {"type": "array", "items": {"type": "string"}},
        "meridian_gotchas": {"type": "array", "items": {"type": "string"}},
        "post_compression_recovery": {"type": "array", "items": {"type": "string"}},
        "anti_patterns_to_avoid": {"type": "array", "items": {"type": "string"}},
        "self_management": {"type": "array", "items": {"type": "string"}},
        "qwen3_models": {"type": "array", "items": {"type": "string"}},
        "decay_freshness_scoring": {"type": "array", "items": {"type": "string"}},
    },
}


IDENTITY_PATTERNS = [
    "who am i", "who are you", "what do we",
    "our relationship", "my personality", "your personality",
    "how do we work", "how do you work", "what are you",
    "tell me about yourself", "tell me about us",
    "what's your name", "your name", "my name",
]

TASK_STATE_PATTERNS = [
    "current task", "what am i working on", "work in progress",
    "what was i doing", "active work", "task state",
    "what are we doing", "where did i leave off",
    "last checkpoint", "session state",
]


def _is_identity_query(query: str) -> bool:
    """Detect self-referential or identity queries."""
    q = query.lower().strip()
    return any(p in q for p in IDENTITY_PATTERNS)


def _is_task_state_query(query: str) -> bool:
    """Detect queries about current task or work in progress."""
    q = query.lower().strip()
    return any(p in q for p in TASK_STATE_PATTERNS)


def _rerank(results: list[dict]) -> list[dict]:
    """Rerank results by blending vector score with importance and freshness.

    Components:
    - importance boost: nonlinear, imp5 gets 3x the boost of imp2
      imp1: +0.01, imp2: +0.03, imp3: +0.05, imp4: +0.08, imp5: +0.12
    - freshness boost: starts at 0.05, decays by 0.001/day (normal) or 0.005/day (volatile)
      Normal:   1d: +0.049, 1w: +0.043, 1mo: +0.02, 50d: +0.0
      Volatile: 1d: +0.045, 3d: +0.035, 1w: +0.015, 10d: +0.0

    Volatile memories (ports, PIDs, temp paths) decay 5x faster — after 10 days
    they get zero freshness boost, making them naturally deprioritized.
    """
    # Nonlinear importance curve — critical memories get a real advantage
    IMPORTANCE_BOOST = {1: 0.01, 2: 0.03, 3: 0.05, 4: 0.08, 5: 0.12}

    now = datetime.now(timezone.utc)
    for r in results:
        importance = r.get("importance", 3)
        importance_boost = IMPORTANCE_BOOST.get(importance, 0.05)

        # Volatile memories decay 5x faster
        is_volatile = r.get("volatile", False)
        decay_rate = 0.005 if is_volatile else 0.001

        # Parse created_at for freshness calculation
        freshness_boost = 0.0
        created_at = r.get("created_at", "")
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                days_old = (now - dt).total_seconds() / 86400
                freshness_boost = max(0.0, 0.05 - (days_old * decay_rate))
            except (ValueError, TypeError):
                pass

        r["score"] = r["score"] + importance_boost + freshness_boost
    return sorted(results, key=lambda r: r["score"], reverse=True)


class MemoryGateway:
    """Synthesis and query routing layer."""

    def __init__(self, ollama_url: str | None = None, model: str | None = None):
        self.ollama_url = ollama_url or config.ollama_url
        self.model = model or config.gateway_model
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=120.0)
        return self._http

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    async def _generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        json_schema: dict | None = None,
    ) -> str:
        """Call the local model via Ollama chat API.

        Args:
            json_schema: If provided, Ollama enforces this JSON schema on output
                         via the 'format' parameter (structured output).
        """
        http = await self._get_http()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or GATEWAY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1 if json_schema else 0.3,
            },
        }
        if json_schema:
            payload["format"] = json_schema
        try:
            resp = await http.post(f"{self.ollama_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            return content
        except Exception as e:
            logger.error(f"Gateway generation failed: {e}")
            return f"[Gateway error: {e}]"

    async def assemble_briefing(self, project_id: str, storage: StorageLayer, agent_id: str | None = None) -> str:
        """Build the hot memory YAML briefing for session start.

        If agent_id is provided, scopes checkpoint retrieval to that agent.
        Other agents' checkpoints are excluded so each agent rehydrates its own state.
        """
        # Gather raw data — scoped to agent if provided
        checkpoint = await storage.get_latest_checkpoint(project_id, source=agent_id)
        decisions = await storage.get_recent_decisions(project_id, limit=5)
        warnings = await storage.get_active_warnings(project_id)

        # If no checkpoint exists, return a minimal briefing
        if not checkpoint and not decisions and not warnings:
            return _minimal_briefing(project_id)

        # Build context for the 30B to synthesize
        raw_context = f"PROJECT: {project_id}\n\n"

        if checkpoint:
            raw_context += f"LATEST CHECKPOINT ({checkpoint['created_at']}):\n"
            raw_context += f"  Task: {checkpoint['task_state']}\n"
            raw_context += f"  Decisions: {', '.join(checkpoint['decisions']) if checkpoint['decisions'] else 'none'}\n"
            raw_context += f"  Warnings: {', '.join(checkpoint['warnings']) if checkpoint['warnings'] else 'none'}\n"
            raw_context += f"  Next steps: {', '.join(checkpoint['next_steps']) if checkpoint['next_steps'] else 'none'}\n"
            raw_context += f"  Working set: {checkpoint['working_set']}\n\n"

        if decisions:
            raw_context += "RECENT DECISIONS:\n"
            for d in decisions:
                raw_context += f"  - [{d['id']}] {d['title']} (importance: {d['importance']}, {d['created_at']})\n"
                raw_context += f"    Rationale: {d['rationale'][:200]}\n"
            raw_context += "\n"

        if warnings:
            raw_context += "ACTIVE WARNINGS:\n"
            for w in warnings:
                raw_context += f"  - [{w['severity'].upper()}] {w['content']}\n"
            raw_context += "\n"

        prompt = f"""Populate a structured briefing from this raw data. Output ONLY valid JSON matching the required schema.

FIELD GUIDE:
- project: project identifier string
- latest_checkpoint.task: 1-2 sentence summary of current work — MOST IMPORTANT FIELD
- latest_checkpoint.decisions: key decisions from the checkpoint (short strings)
- latest_checkpoint.warnings: ONLY blockers/threats to current work, prefix each [HIGH], [MED], or [LOW]
- latest_checkpoint.next_steps: what to do next (from checkpoint data)
- latest_checkpoint.working_set: files and endpoints currently relevant
- recent_decisions: top 5 decisions with id, description, and rationale from raw data
- active_warnings: all warnings from raw data, prefix each [HIGH], [MED], or [LOW]
- meridian_gotchas: technical gotchas (API quirks, version issues, config traps)

RULES:
1. Transfer ALL data from raw input — don't drop or summarize away important details.
2. Each string should be 1-2 sentences max. Be concise but complete.
3. Map raw severity levels directly: HIGH/CRITICAL -> [HIGH], MEDIUM -> [MED], LOW -> [LOW].
4. For recent_decisions, preserve the original memory ID in the id field.

RAW DATA:
{raw_context}"""

        raw_json = await self._generate(
            prompt,
            max_tokens=config.hot_memory_max_tokens,
            json_schema=BRIEFING_SCHEMA,
        )

        # Parse structured JSON output and convert to YAML
        if not raw_json or raw_json.startswith("[Gateway error"):
            return _fallback_briefing(project_id, checkpoint, decisions, warnings)

        try:
            briefing_data = json.loads(raw_json)
            briefing_yaml = yaml.dump(briefing_data, default_flow_style=False, sort_keys=False, allow_unicode=True)
            return f"# HOT MEMORY — auto-generated by Meridian Gateway\n```yaml\n{briefing_yaml}```"
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            logger.warning(f"Failed to parse structured briefing output: {e}")
            # Fall back to raw output if JSON parsing fails
            return f"# HOT MEMORY — auto-generated by Meridian Gateway\n```yaml\n{raw_json}\n```"

    async def recall(
        self,
        query: str,
        scope: str,
        max_tokens: int,
        storage: StorageLayer,
        agent_id: str | None = None,
    ) -> dict:
        """Semantic search + synthesis. Returns compressed, relevant summary.

        Routing logic:
        1. Identity queries → force personality/identity tag filter
        2. Tag matches in query → filter Qdrant to those tags
        3. No matches → pure vector search
        Results are reranked by importance before synthesis.
        """
        # Inject checkpoint context for task-state queries
        checkpoint_context = ""
        if _is_task_state_query(query):
            checkpoint = await storage.get_latest_checkpoint(source=agent_id)
            if checkpoint:
                checkpoint_context = (
                    f"\nLATEST CHECKPOINT ({checkpoint['created_at']}):\n"
                    f"  Task: {checkpoint['task_state']}\n"
                    f"  Next steps: {', '.join(checkpoint['next_steps']) if checkpoint['next_steps'] else 'none'}\n"
                    f"  Working set: {checkpoint['working_set']}\n"
                )

        # Route identity queries to personality memories
        if _is_identity_query(query):
            matched_tags = ["personality", "identity"]
        else:
            # Check if query matches any known tags
            known_tags = await storage.get_all_tags()
            matched_tags = storage._extract_tag_matches(query, known_tags)

        # Try tag-filtered search first
        tag_filter = matched_tags if matched_tags else None
        raw_results = await storage.search_memories(
            query, limit=10, scope=scope, tag_filter=tag_filter,
        )

        # If tag filter returned too few results, fall back to unfiltered
        if tag_filter and len(raw_results) < 3:
            unfiltered = await storage.search_memories(query, limit=10, scope=scope)
            # Merge: dedupe by id, tag-filtered results first
            seen = {r["id"] for r in raw_results}
            for r in unfiltered:
                if r["id"] not in seen:
                    raw_results.append(r)
                    seen.add(r["id"])
            raw_results = raw_results[:10]

        # If warm search found too few results, try cold storage fallback
        if len(raw_results) < 3:
            try:
                cold_results = await storage.search_cold(query, limit=5)
                if cold_results:
                    # Merge cold results (they don't have vector scores, use 0.5 as baseline)
                    seen = {r["id"] for r in raw_results}
                    for cr in cold_results:
                        if cr["id"] not in seen:
                            cr["score"] = 0.5  # Baseline score for cold results
                            raw_results.append(cr)
                            seen.add(cr["id"])
                    logger.info(f"Cold fallback: added {len(cold_results)} results for '{query[:40]}'")
            except Exception as e:
                logger.warning(f"Cold storage fallback failed: {e}")

        if not raw_results:
            return {
                "results": f"No memories found matching '{query}' in scope '{scope}'.",
                "sources": [],
                "token_count": 0,
            }

        # Rerank by importance + freshness (breaks ties when vector scores are close)
        raw_results = _rerank(raw_results)

        # Build context for synthesis
        tag_note = f"\nTAG FILTER: {', '.join(matched_tags)}" if matched_tags else ""
        context = f"QUERY: {query}\nSCOPE: {scope}{tag_note}\n"
        if checkpoint_context:
            context += f"\n{checkpoint_context}\n"
        context += "\nSEARCH RESULTS (newest first):\n"
        for r in raw_results:
            source_tag = f", source: {r['source']}" if r.get('source') else ""
            # Include age for temporal reasoning
            age_str = ""
            created = r.get("created_at", "")
            if created:
                try:
                    dt = datetime.fromisoformat(created)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    hours_old = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                    if hours_old < 1:
                        age_str = ", age: <1h"
                    elif hours_old < 24:
                        age_str = f", age: {int(hours_old)}h"
                    else:
                        age_str = f", age: {int(hours_old/24)}d"
                except (ValueError, TypeError):
                    pass
            volatile_tag = ", VOLATILE" if r.get("volatile") else ""
            cold_tag = ", FROM COLD ARCHIVE" if r.get("cold") else ""
            context += f"\n[{r['id']}] (type: {r['type']}, importance: {r['importance']}{source_tag}{age_str}{volatile_tag}{cold_tag})\n"
            context += f"  {r['content']}\n"
            if r['tags']:
                context += f"  Tags: {', '.join(r['tags'])}\n"

        prompt = f"""Synthesize these search results into a direct answer to the query.

RULES:
1. RUTHLESSLY filter. If a result doesn't DIRECTLY answer the query, DROP IT. Only ~30-50% of results will be relevant — that's expected.
2. EVERY fact MUST have an inline citation using the memory ID from the results. Example: "The server runs on port 8080 [mem_abc123]." If you can't cite it, don't say it. Use the exact [mem_xxx] IDs from the search results below.
3. Preserve rationale: "Chose X because Y" not just "Uses X."
4. Conflicts: prefer newest (check age field). Flag: "[mem_old] (Xd) says Y but newer [mem_new] (<1h) says Z — using newer."
5. Lead with the answer. Details after.
6. Budget: {max_tokens} tokens MAX. Stop when the query is answered. Don't pad.
7. Write flowing prose. NO section headers, NO bullet lists unless the query asks for a list.
8. NEVER invent facts not present in the search results below.
9. VOLATILE results (ports, PIDs, temp paths) decay fast and may be stale. If citing a volatile result older than a few days, note it may have changed.
10. Importance scale: 5=critical, 4=significant, 3=normal, 2=minor, 1=trivial. Strongly prefer importance 4-5 results. Only include importance 1-2 if they directly answer the query.
11. If source attribution is present (source: giga, webbie, chris), note who said it when it matters for context.

{context}

Answer:"""

        summary = await self._generate(prompt, max_tokens=max_tokens)

        sources = [
            {"id": r["id"], "type": r["type"], "relevance": r["score"], "created_at": r["created_at"]}
            for r in raw_results[:5]
        ]

        # Auto-promote cold results that were used in synthesis
        cold_results_used = [r for r in raw_results[:5] if r.get("cold")]
        for cr in cold_results_used:
            try:
                promoted = await storage.promote_from_cold(cr["id"])
                if promoted:
                    logger.info(f"Auto-promoted {cr['id']} from cold to warm")
            except Exception as e:
                logger.warning(f"Auto-promotion failed for {cr['id']}: {e}")

        return {
            "results": summary,
            "sources": sources,
            "token_count": len(summary.split()),
        }

    async def graph_query(self, args: dict, storage: StorageLayer) -> dict:
        """Traverse entity graph and synthesize results."""
        entity = args["entity"]
        relation = args.get("relation")
        depth = args.get("depth", 2)

        raw_graph = await storage.query_graph(entity, relation, depth)

        if not raw_graph["entities"]:
            return {
                "entities": [],
                "summary": f"No entity found matching '{entity}'.",
            }

        # Build context for 30B synthesis
        context = f"ENTITY GRAPH starting from '{entity}':\n\n"
        context += "ENTITIES:\n"
        for e in raw_graph["entities"]:
            context += f"  - {e['name']} (type: {e['type']})\n"
        context += "\nRELATIONS:\n"
        for r in raw_graph["relations"]:
            context += f"  - {r['source']} --{r['type']}--> {r['target']}\n"

        prompt = f"""Explain this entity relationship graph. Focus on how components connect and what depends on what.

Relation types: calls (invokes/uses), depends_on (requires), configured_by (settings from), implements (realizes), broke (caused failure in).

RULES:
1. Lead with the queried entity and its direct connections.
2. Note critical dependencies — what breaks if this entity goes down?
3. Flowing prose, 2-4 sentences. No bullet lists.
4. If the graph is sparse (few relations), say so — don't invent connections.

{context}

Summary:"""

        summary = await self._generate(prompt, max_tokens=500)

        return {
            "entities": raw_graph["entities"],
            "relations": raw_graph["relations"],
            "summary": summary,
        }

    async def history_search(self, args: dict, storage: StorageLayer, workers=None) -> dict:
        """Search episodic cold storage — keyword search on indexes + vector search on ingested JSONL."""
        query = args["query"]

        # Path 1: keyword search on episode index files (legacy)
        episodes = await storage.search_episodes(query, limit=3)
        episode_results = [
            {
                "session_id": ep["session_id"],
                "started_at": ep.get("started_at", ""),
                "summary": ep.get("summary", ""),
                "relevance": 1.0,
                "source": "episode_index",
            }
            for ep in episodes
        ]

        # Path 2: vector search on ingested JSONL conversation chunks
        vector_results = await storage.search_memories(
            query, limit=5, scope="all", source_filter="episodic_ingest"
        )
        conversation_results = [
            {
                "session_id": r.get("id", "").split("_")[1] if "_" in r.get("id", "") else "unknown",
                "content": r.get("content", "")[:500],
                "relevance": r.get("score", 0),
                "source": "episodic_vector",
            }
            for r in vector_results
        ]

        all_results = episode_results + conversation_results
        token_count = sum(
            len(r.get("summary", r.get("content", "")).split()) for r in all_results
        )

        return {
            "episodes": episode_results,
            "conversations": conversation_results,
            "token_count": token_count,
        }


def _minimal_briefing(project_id: str) -> str:
    return f"""# HOT MEMORY — auto-generated by Meridian Gateway
identity:
  project: "{project_id}"
  role: "Software developer with persistent memory"
  style: "Direct, technical"

task:
  current: "No previous session found — this is a fresh start."
  blocked_on: null
  last_checkpoint: null
  context: "First session for this project. Use memory_remember to store important decisions."

decisions: []

working_set:
  files: []
  endpoints: []

warnings: []
"""


def _fallback_briefing(project_id: str, checkpoint: dict | None, decisions: list, warnings: list) -> str:
    """Structured fallback when Gateway 30B is unavailable."""
    lines = [
        "# HOT MEMORY — fallback (Gateway unavailable)",
        f"identity:",
        f'  project: "{project_id}"',
        f'  role: "Software developer with persistent memory"',
        f'  style: "Direct, technical"',
        "",
        "task:",
    ]
    if checkpoint:
        lines.append(f'  current: "{checkpoint["task_state"]}"')
        lines.append(f'  last_checkpoint: "{checkpoint["created_at"]}"')
        if checkpoint.get("next_steps"):
            lines.append(f'  context: "Next steps: {"; ".join(checkpoint["next_steps"])}"')
    else:
        lines.append('  current: "Unknown — no checkpoint found"')

    lines.append("")
    lines.append("decisions:")
    for d in decisions[:5]:
        lines.append(f'  - "{d["title"][:80]} ({d["created_at"]})"')

    lines.append("")
    lines.append("warnings:")
    for w in warnings[:5]:
        lines.append(f'  - "[{w["severity"].upper()}] {w["content"]}"')

    lines.append("")
    lines.append("working_set:")
    if checkpoint and checkpoint.get("working_set"):
        ws = checkpoint["working_set"]
        lines.append(f"  files: {ws.get('files', [])}")
        lines.append(f"  endpoints: {ws.get('endpoints', [])}")
    else:
        lines.append("  files: []")

    return "\n".join(lines) + "\n"
