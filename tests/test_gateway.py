"""Test the Memory Gateway — 30B synthesis layer."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridian.storage import StorageLayer
from meridian.gateway import MemoryGateway


async def main():
    print("=== Testing Memory Gateway ===\n")

    storage = StorageLayer()
    await storage.init_db()
    gateway = MemoryGateway()

    # Test briefing assembly
    print("Assembling hot memory briefing...")
    briefing = await gateway.assemble_briefing("default", storage)
    print(f"[OK] Briefing ({len(briefing)} chars, ~{len(briefing.split())} tokens):")
    print("-" * 60)
    print(briefing[:1500])
    print("-" * 60)

    # Test recall (semantic search + synthesis)
    print("\nTesting memory recall...")
    result = await gateway.recall(
        query="WebSocket issues and bugs",
        scope="all",
        max_tokens=500,
        storage=storage,
    )
    print(f"[OK] Recall result ({result['token_count']} tokens, {len(result['sources'])} sources):")
    print(result["results"][:500])

    # Test graph query
    print("\nTesting graph query...")
    graph_result = await gateway.graph_query(
        {"entity": "FastAPI", "depth": 2},
        storage,
    )
    print(f"[OK] Graph: {len(graph_result['entities'])} entities")
    print(f"Summary: {graph_result['summary'][:300]}")

    await storage.close()
    await gateway.close()
    print("\n=== Gateway tests passed! ===")


if __name__ == "__main__":
    asyncio.run(main())
