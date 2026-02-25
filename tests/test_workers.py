"""Test the 8B Worker Pool — entity extraction, summarization, tagging."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridian.storage import StorageLayer
from meridian.workers import WorkerPool


async def main():
    print("=== Testing Worker Pool ===\n")

    storage = StorageLayer()
    await storage.init_db()
    workers = WorkerPool()

    # Test entity extraction
    print("Testing entity extraction...")
    content = "We switched from PostgreSQL to Qdrant for vector storage because Qdrant has better HNSW indexing. The StorageLayer in storage.py wraps both Qdrant and SQLite."
    await workers.extract_and_store_entities(content, storage)
    print("[OK] Entity extraction complete")

    # Check what was stored
    graph = await storage.query_graph("Qdrant", depth=2)
    print(f"  Entities: {len(graph['entities'])}, Relations: {len(graph['relations'])}")
    for e in graph['entities']:
        print(f"  - {e['name']} ({e['type']})")
    for r in graph['relations']:
        print(f"  - {r['source']} --{r['type']}--> {r['target']}")

    # Test tag classification
    print("\nTesting tag classification...")
    tags = await workers.classify_tags("We chose FastAPI for the web server because of async support and WebSocket handling")
    print(f"[OK] Tags: {tags}")

    # Test session summarization
    print("\nTesting session summarization...")
    messages = [
        {"role": "user", "content": "Let's implement the memory storage layer"},
        {"role": "assistant", "content": "I'll build the StorageLayer class that wraps Qdrant for vectors and SQLite for the knowledge graph. Let me start with the embedding pipeline."},
        {"role": "user", "content": "The embedding model is nomic-embed-text on Ollama"},
        {"role": "assistant", "content": "Got it. I'll use the Ollama embed API with nomic-embed-text. The vector dimension is 768. Let me also set up the Qdrant collection with cosine distance."},
        {"role": "user", "content": "Great, also add SQLite for the entity graph"},
        {"role": "assistant", "content": "Done. I've implemented entities, relations, decisions, warnings, and checkpoints tables in SQLite. The StorageLayer now has full CRUD for both stores plus episodic JSON storage."},
    ]
    summary = await workers.summarize_session(messages)
    print(f"[OK] Summary: {summary.get('summary', 'N/A')}")
    print(f"  Key events: {len(summary.get('key_events', []))}")
    for event in summary.get("key_events", []):
        print(f"  - [{event.get('type', '?')}] {event.get('content', '?')[:80]}")

    await storage.close()
    await workers.close()
    print("\n=== Worker tests passed! ===")


if __name__ == "__main__":
    asyncio.run(main())
