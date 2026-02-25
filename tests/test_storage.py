"""Test the StorageLayer — Qdrant + SQLite + Episodic store."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridian.storage import StorageLayer
from meridian.schemas import MemoryRecord, Checkpoint, Entity, Relation, Decision, Warning


async def main():
    print("=== Testing StorageLayer ===\n")

    storage = StorageLayer()
    await storage.init_db()
    print("[OK] SQLite initialized")

    # Test embedding
    print("\nTesting embedding...")
    vec = await storage.embed("This is a test of the embedding system")
    print(f"[OK] Embedding generated: dim={len(vec)}, first 5 values: {vec[:5]}")

    # Test store memory
    print("\nStoring test memories...")
    mem1 = MemoryRecord(
        content="We chose FastAPI for the web server because of native async support and WebSocket handling.",
        type="decision",
        tags=["fastapi", "web", "architecture"],
        importance=4,
        project_id="default",
    )
    id1 = await storage.store_memory(mem1)
    print(f"[OK] Stored decision: {id1}")

    mem2 = MemoryRecord(
        content="Don't use asyncio.gather for concurrent WebSocket connections — causes race condition with shared state.",
        type="warning",
        tags=["asyncio", "websocket", "bug"],
        importance=5,
        project_id="default",
    )
    id2 = await storage.store_memory(mem2)
    print(f"[OK] Stored warning: {id2}")

    mem3 = MemoryRecord(
        content="Always validate JSON responses from Ollama before parsing — the 8B model sometimes wraps in markdown code blocks.",
        type="pattern",
        tags=["ollama", "json", "parsing"],
        importance=3,
        project_id="default",
    )
    id3 = await storage.store_memory(mem3)
    print(f"[OK] Stored pattern: {id3}")

    # Test search
    print("\nSearching memories...")
    results = await storage.search_memories("WebSocket race condition bug", limit=3)
    print(f"[OK] Search returned {len(results)} results:")
    for r in results:
        print(f"  - [{r['type']}] score={r['score']:.3f}: {r['content'][:80]}...")

    # Test scoped search
    results = await storage.search_memories("architecture", scope="decisions", limit=3)
    print(f"\n[OK] Scoped search (decisions): {len(results)} results")

    # Test entity storage
    print("\nStoring entities...")
    ent1 = Entity(name="FastAPI", type="service", project_id="default")
    ent2 = Entity(name="Qdrant", type="service", project_id="default")
    ent3 = Entity(name="Ollama", type="service", project_id="default")
    e1 = await storage.store_entity(ent1)
    e2 = await storage.store_entity(ent2)
    e3 = await storage.store_entity(ent3)
    print(f"[OK] Stored entities: {e1}, {e2}, {e3}")

    # Store relations
    rel1 = Relation(source_id=ent1.id, target_id=ent2.id, type="calls")
    rel2 = Relation(source_id=ent1.id, target_id=ent3.id, type="calls")
    await storage.store_relation(rel1)
    await storage.store_relation(rel2)
    print("[OK] Stored relations")

    # Test graph query
    print("\nQuerying graph...")
    graph = await storage.query_graph("FastAPI", depth=2)
    print(f"[OK] Graph: {len(graph['entities'])} entities, {len(graph['relations'])} relations")
    for r in graph['relations']:
        print(f"  {r['source']} --{r['type']}--> {r['target']}")

    # Test decisions
    print("\nStoring decisions...")
    dec = Decision(
        title="Use Qdrant for vector storage",
        rationale="Open source, fast, good Python client, runs locally",
        tags=["qdrant", "vector", "storage"],
        importance=4,
        project_id="default",
    )
    await storage.store_decision(dec)
    decisions = await storage.get_recent_decisions("default")
    print(f"[OK] {len(decisions)} recent decisions")

    # Test warnings
    print("\nStoring warnings...")
    wrn = Warning(
        content="Qdrant binary needs --config-path, not --storage-path",
        severity="medium",
        project_id="default",
    )
    await storage.store_warning(wrn)
    warnings = await storage.get_active_warnings("default")
    print(f"[OK] {len(warnings)} active warnings")

    # Test checkpoint
    print("\nStoring checkpoint...")
    cp_result = await storage.checkpoint({
        "task_state": "Testing Meridian storage layer",
        "decisions": ["Use Qdrant for vectors", "Use SQLite for graph"],
        "warnings": ["Qdrant CLI flag gotcha"],
        "next_steps": ["Test Gateway", "Test Workers", "Build Web UI"],
        "working_set": {
            "files": ["meridian/storage.py", "meridian/schemas.py"],
            "endpoints": ["/ws/{project_id}", "/api/health"],
        },
    })
    print(f"[OK] Checkpoint stored: {cp_result}")

    latest_cp = await storage.get_latest_checkpoint("default")
    print(f"[OK] Latest checkpoint: {latest_cp['task_state']}")

    # Test memory count
    count = await storage.get_memory_count()
    print(f"\n[OK] Total memories in Qdrant: {count}")

    # Test project status
    status = await storage.project_status("default")
    print(f"[OK] Project status: {status}")

    await storage.close()
    print("\n=== All storage tests passed! ===")


if __name__ == "__main__":
    asyncio.run(main())
