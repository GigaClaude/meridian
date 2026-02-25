"""Integration test — full Meridian lifecycle using storage + gateway directly."""

import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridian.schemas import MemoryRecord
from meridian.storage import StorageLayer
from meridian.gateway import MemoryGateway
from meridian.workers import WorkerPool
from meridian.session import SessionManager


async def main():
    print("=== Integration Test: Full Meridian Lifecycle ===\n")

    # 1. Init subsystems
    storage = StorageLayer()
    await storage.init_db()
    gateway = MemoryGateway()
    workers = WorkerPool()
    print("[1] Subsystems initialized")

    # 2. Boot session
    session = SessionManager(
        project_id="default",
        gateway=gateway,
        storage=storage,
        workers=workers,
    )
    system_prompt = await session.boot()
    print(f"[2] Session booted: {session.session_id}")
    print(f"    System prompt: {len(system_prompt)} chars")

    # 3. Test memory operations directly via storage
    print("\n[3] Testing memory operations...")

    # Store a decision
    mem = MemoryRecord(
        content="Switched to qwen2.5-coder:14b for Gateway because qwen3 models have thinking mode that consumes all tokens.",
        type="decision",
        tags=["ollama", "model-selection", "gateway"],
        importance=4,
        project_id="default",
    )
    mem_id = await storage.store_memory(mem)
    print(f"  Stored decision: {mem_id}")

    # Recall via gateway (semantic search + synthesis)
    result = await gateway.recall(
        query="What model should I use for the Gateway?",
        scope="decisions",
        max_tokens=400,
        storage=storage,
    )
    print(f"  Recall ({result['token_count']} tokens, {len(result['sources'])} sources):")
    print(f"  {result['results'][:200]}...")

    # Checkpoint
    cp_result = await storage.checkpoint({
        "task_state": "Integration testing complete. All subsystems operational.",
        "decisions": [
            "Using qwen2.5-coder:14b for Gateway and Workers",
            "nomic-embed-text for 768-dim embeddings",
        ],
        "warnings": [
            "qwen3 models have thinking mode — use qwen2.5 series instead",
        ],
        "next_steps": [
            "Start FastAPI server and test WebSocket chat",
        ],
        "working_set": {
            "files": ["meridian/storage.py", "meridian/gateway.py"],
            "endpoints": ["/ws/{project_id}", "/api/health"],
        },
    })
    print(f"  Checkpoint: {cp_result}")

    # Graph query
    graph_result = await gateway.graph_query(
        {"entity": "StorageLayer", "depth": 2},
        storage,
    )
    print(f"  Graph: {len(graph_result['entities'])} entities")
    print(f"  Summary: {graph_result['summary'][:200]}...")

    # Re-assemble briefing (should now include checkpoint)
    print("\n[4] Re-assembling briefing with new data...")
    briefing = await gateway.assemble_briefing("default", storage)
    print(f"  Briefing ({len(briefing)} chars):")
    print(f"  {briefing[:500]}...")

    # 5. End session
    print("\n[5] Ending session...")
    await session.end()

    # Verify data persisted
    status = await storage.project_status("default")
    print(f"\nFinal project status: {status}")

    await storage.close()
    await gateway.close()
    await workers.close()
    print("\n=== Integration test complete! ===")


if __name__ == "__main__":
    asyncio.run(main())
