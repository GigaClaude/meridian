#!/usr/bin/env python3
"""Ingest Claude Code JSONL conversation logs into Meridian's episodic memory.

Reads JSONL files from ~/.claude/projects/, chunks by turn pairs (human + assistant),
embeds via Ollama, and stores in Qdrant with source=episodic tagging.

Architecture by Webbie. Built by GigaClaude. Feb 26, 2026.

Usage:
    python ingest_jsonl.py                    # Ingest all unprocessed sessions
    python ingest_jsonl.py --file UUID.jsonl  # Ingest one specific file
    python ingest_jsonl.py --dry-run          # Show what would be ingested
    python ingest_jsonl.py --stats            # Show session statistics
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Meridian imports — add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.config import config
from meridian.storage import StorageLayer

JSONL_DIR = Path.home() / ".claude" / "projects" / "-home-claude"
STATE_FILE = config.data_dir / "jsonl_ingest_state.json"

# Chunking config
MAX_CHUNK_TOKENS = 400   # Rough token estimate per chunk — stay under mxbai-embed-large 512 ctx
MIN_CHUNK_CHARS = 100    # Skip tiny chunks
MAX_CONTENT_CHARS = 1500 # Truncate very long content blocks (mxbai ctx = 512 tokens ≈ 2048 chars)
MAX_COMBINED_CHARS = 1800 # Hard cap on final combined chunk (~450 tokens)


def load_state() -> dict:
    """Load ingestion state — tracks which files have been processed."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"processed": {}, "last_run": None}


def save_state(state: dict):
    """Save ingestion state."""
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def file_hash(path: Path) -> str:
    """Quick hash of file size + mtime for change detection."""
    stat = path.stat()
    return hashlib.md5(f"{stat.st_size}:{stat.st_mtime}".encode()).hexdigest()


def extract_text(content) -> str:
    """Extract readable text from a message content field.

    Content can be:
    - A string (user messages)
    - A list of blocks (assistant messages with thinking, tool_use, text)
    - A dict with 'content' key
    """
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        # Nested message format: {role, content}
        return extract_text(content.get("content", ""))

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    name = block.get("name", "?")
                    inp = str(block.get("input", ""))[:200]
                    parts.append(f"[tool: {name}({inp})]")
                elif btype == "tool_result":
                    # Include brief tool results
                    result = str(block.get("content", ""))[:200]
                    parts.append(f"[result: {result}]")
                # Skip thinking blocks, signatures — they're noise
        return " ".join(parts)

    return str(content)[:MAX_CONTENT_CHARS]


def parse_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of turn records."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def chunk_session(records: list[dict], session_id: str) -> list[dict]:
    """Chunk a session's records into embeddable turn pairs.

    Strategy: pair each user message with the following assistant response(s).
    Skip queue-operation records. Merge consecutive assistant messages.
    """
    # Filter to user and assistant messages only
    messages = []
    for r in records:
        rtype = r.get("type", "")
        if rtype not in ("user", "assistant"):
            continue

        msg = r.get("message", r)
        role = msg.get("role", rtype)
        content = msg.get("content", r.get("content", ""))
        ts = r.get("timestamp", "")
        text = extract_text(content)

        if text and len(text.strip()) > 0:
            messages.append({"role": role, "text": text.strip(), "ts": ts})

    # Pair into chunks: user + following assistant responses
    chunks = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "user":
            user_text = msg["text"][:MAX_CONTENT_CHARS]
            ts = msg["ts"]

            # Collect following assistant responses
            assistant_parts = []
            j = i + 1
            while j < len(messages) and messages[j]["role"] == "assistant":
                assistant_parts.append(messages[j]["text"][:MAX_CONTENT_CHARS])
                j += 1

            if assistant_parts:
                assistant_text = " ".join(assistant_parts)[:MAX_CONTENT_CHARS]
                combined = f"Human: {user_text}\n\nAssistant: {assistant_text}"
                combined = combined[:MAX_COMBINED_CHARS]

                if len(combined) >= MIN_CHUNK_CHARS:
                    chunk_id = f"ep_{session_id[:12]}_{len(chunks):04d}"
                    chunks.append({
                        "id": chunk_id,
                        "session_id": session_id,
                        "content": combined,
                        "timestamp": ts,
                        "turn_index": len(chunks),
                    })
                i = j
            else:
                i += 1
        else:
            # Orphan assistant message (no preceding user)
            i += 1

    return chunks


def get_sessions(specific_file: str | None = None) -> list[tuple[str, Path]]:
    """Get list of (session_id, path) tuples for JSONL files."""
    if not JSONL_DIR.exists():
        print(f"JSONL directory not found: {JSONL_DIR}", file=sys.stderr)
        return []

    sessions = []
    for path in sorted(JSONL_DIR.glob("*.jsonl")):
        session_id = path.stem  # UUID filename without .jsonl
        if specific_file and specific_file not in (path.name, session_id):
            continue
        sessions.append((session_id, path))

    return sessions


async def ingest_chunks(storage: StorageLayer, chunks: list[dict], dry_run: bool = False) -> int:
    """Embed and store chunks in Qdrant. Returns count of stored chunks."""
    stored = 0
    for chunk in chunks:
        if dry_run:
            print(f"  [{chunk['id']}] {chunk['content'][:80]}...")
            stored += 1
            continue

        # Embed
        vector = await storage.embed(chunk["content"])

        # Store in Qdrant with episodic tagging
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"]))
        from qdrant_client.models import PointStruct
        point = PointStruct(
            id=point_uuid,
            vector=vector,
            payload={
                "mem_id": chunk["id"],
                "content": chunk["content"],
                "type": "episodic",
                "tags": ["episodic", "jsonl", "conversation"],
                "project_id": config.project_id,
                "importance": 1,  # Low importance — raw conversation
                "source": "episodic_ingest",
                "volatile": False,
                "created_at": chunk.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "superseded_by": None,
                "session_id": chunk["session_id"],
                "related_ids": [],
                "turn_index": chunk["turn_index"],
            },
        )
        storage.qdrant.upsert(
            collection_name=config.qdrant_collection,
            points=[point],
        )
        stored += 1

    return stored


async def run_stats():
    """Show statistics about available JSONL sessions."""
    sessions = get_sessions()
    state = load_state()

    print(f"JSONL directory: {JSONL_DIR}")
    print(f"Sessions found:  {len(sessions)}")
    print(f"Already ingested: {len(state.get('processed', {}))}")
    print()

    total_chunks = 0
    total_bytes = 0
    for session_id, path in sessions:
        size = path.stat().st_size
        total_bytes += size
        records = parse_jsonl(path)
        chunks = chunk_session(records, session_id)
        status = "DONE" if session_id in state.get("processed", {}) else "NEW"
        print(f"  {session_id[:12]}... {size:>10,} bytes  {len(chunks):>4} chunks  [{status}]")
        total_chunks += len(chunks)

    print(f"\nTotal: {total_bytes:,} bytes, {total_chunks} chunks")


async def run_ingest(specific_file: str | None = None, dry_run: bool = False):
    """Main ingestion loop."""
    sessions = get_sessions(specific_file)
    state = load_state()

    if not sessions:
        print("No JSONL files found to ingest.")
        return

    # Filter to unprocessed (or changed) files
    to_process = []
    for session_id, path in sessions:
        fh = file_hash(path)
        prev = state.get("processed", {}).get(session_id, {})
        if prev.get("hash") == fh and not specific_file:
            continue  # Already processed, unchanged
        to_process.append((session_id, path, fh))

    if not to_process:
        print("All sessions already ingested. Use --file to force re-ingestion.")
        return

    print(f"Sessions to ingest: {len(to_process)}")
    if dry_run:
        print("(DRY RUN — nothing will be stored)\n")

    # Init storage
    storage = None
    if not dry_run:
        storage = StorageLayer()

    total_chunks = 0
    total_stored = 0
    t0 = time.time()

    for session_id, path, fh in to_process:
        records = parse_jsonl(path)
        chunks = chunk_session(records, session_id)

        if not chunks:
            print(f"  {session_id[:12]}... 0 chunks (skipped)")
            continue

        print(f"  {session_id[:12]}... {len(chunks)} chunks", end="")
        stored = await ingest_chunks(storage, chunks, dry_run=dry_run)
        print(f" -> {stored} stored")

        total_chunks += len(chunks)
        total_stored += stored

        if not dry_run:
            state.setdefault("processed", {})[session_id] = {
                "hash": fh,
                "chunks": len(chunks),
                "stored": stored,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }

    elapsed = time.time() - t0

    if not dry_run:
        save_state(state)

    print(f"\nDone: {total_stored}/{total_chunks} chunks stored in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Claude Code JSONL conversation logs into Meridian",
    )
    parser.add_argument("--file", "-f", help="Specific JSONL file to ingest")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be ingested")
    parser.add_argument("--stats", "-s", action="store_true", help="Show session statistics")

    args = parser.parse_args()

    if args.stats:
        asyncio.run(run_stats())
    else:
        asyncio.run(run_ingest(args.file, args.dry_run))


if __name__ == "__main__":
    main()
