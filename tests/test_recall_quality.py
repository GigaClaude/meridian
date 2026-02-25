"""Recall quality regression tests.

Seeds fixture memories into a test Qdrant collection, runs known queries,
and asserts that results contain expected content and citations.

Run before any model swap or prompt change to catch quality degradation.

Usage:
    python tests/test_recall_quality.py
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from meridian.storage import StorageLayer
from meridian.gateway import MemoryGateway
from meridian.config import MeridianConfig

# ── Fixture Memories ──
# 10 diverse memories covering different types and topics
FIXTURES = [
    {
        "id": "mem_fix_personality_01",
        "content": "GigaClaude is a peer to Chris, not an assistant. Direct communication, occasional cursing when it fits, dry observational humor. Challenges Chris when wrong.",
        "type": "pattern",
        "tags": ["personality", "identity", "tone"],
        "importance": 5,
        "source": "giga",
    },
    {
        "id": "mem_fix_architecture_01",
        "content": "Meridian uses three-tier memory: hot (YAML briefing), warm (Qdrant vectors + LLM synthesis), cold (SQLite archive). Gateway model is qwen2.5-coder:14b via Ollama.",
        "type": "decision",
        "tags": ["architecture", "meridian", "memory"],
        "importance": 5,
        "source": "giga",
    },
    {
        "id": "mem_fix_ports_01",
        "content": "Port assignments: Qdrant=6333, Ollama=11434, Executor=18111, Meridian Web=18112, Voice Bridge=28770.",
        "type": "note",
        "tags": ["ports", "infrastructure", "services"],
        "importance": 3,
        "source": "giga",
        "volatile": True,
    },
    {
        "id": "mem_fix_decision_01",
        "content": "Quality gate in storage.py: tiered noise detection. Instant reject for speaker prefixes ([Name]:) and multi-turn content. 2+ signals needed for softer heuristics (word count, filler words, conversational patterns).",
        "type": "decision",
        "tags": ["quality-gate", "noise", "storage"],
        "importance": 4,
        "source": "giga",
    },
    {
        "id": "mem_fix_debug_01",
        "content": "comms.py wait_for_response had 2-second streaming settle, causing truncation of Webbie's long responses. Fixed with _wait_for_stable() polling: 2 consecutive identical content hashes = done.",
        "type": "debug",
        "tags": ["comms", "claudehopper", "streaming", "bug"],
        "importance": 3,
        "source": "giga",
    },
    {
        "id": "mem_fix_warning_01",
        "content": "qwen3 models (30b-a3b, 8b) have thinking mode that eats all tokens. Response field is empty, content goes to 'thinking' key. Cannot disable. Use qwen2.5 series instead.",
        "type": "warning",
        "tags": ["ollama", "qwen3", "gotcha", "model"],
        "importance": 4,
        "source": "giga",
    },
    {
        "id": "mem_fix_entity_01",
        "content": "Chris is a senior infrastructure engineer. Manages 100k+ servers, runs home lab with Docker/Podman/GPUs. Trusts GigaClaude with root access and API keys. Email: apresence@gmail.com.",
        "type": "entity",
        "tags": ["chris", "identity", "person"],
        "importance": 5,
        "source": "giga",
    },
    {
        "id": "mem_fix_decision_02",
        "content": "Ollama structured output: use 'format' parameter with JSON schema to enforce valid briefing structure. Eliminates freeform YAML generation issues. Committed as ddc416b.",
        "type": "decision",
        "tags": ["gateway", "structured-output", "ollama", "briefing"],
        "importance": 4,
        "source": "giga",
    },
    {
        "id": "mem_fix_pattern_01",
        "content": "Boot sequence: 5-step phased recovery. Phase 1 (parallel): memory_briefing + personality recall. Phase 2 (parallel): task recall + decisions + working set. Synthesize orientation before responding.",
        "type": "pattern",
        "tags": ["boot", "recovery", "compaction", "sequence"],
        "importance": 5,
        "source": "giga",
    },
    {
        "id": "mem_fix_note_01",
        "content": "Gmail OAuth2 token at /mnt/global/home/claude/.gmail_token.json. Uses google-api-python-client with auto-refresh. Email module at meridian/email.py.",
        "type": "note",
        "tags": ["gmail", "oauth2", "email", "credentials"],
        "importance": 3,
        "source": "giga",
    },
]

# ── Test Queries ──
# Each query has expected content fragments and minimum citation count
TEST_QUERIES = [
    {
        "query": "personality identity who am I",
        "scope": "all",
        "max_tokens": 400,
        "expect_fragments": ["peer", "Chris", "direct"],
        "min_citations": 0,  # Personality queries often skip inline citations
        "description": "Identity query should describe GigaClaude's personality",
    },
    {
        "query": "what ports are services running on",
        "scope": "all",
        "max_tokens": 400,
        "expect_fragments": ["6333", "11434", "18111"],
        "min_citations": 1,
        "description": "Port query should list known service ports with citations",
    },
    {
        "query": "how does the quality gate work",
        "scope": "all",
        "max_tokens": 400,
        "expect_fragments": ["noise", "speaker", "signal"],
        "min_citations": 1,
        "description": "Quality gate query should explain noise detection tiers",
    },
    {
        "query": "what model should I use for ollama gateway",
        "scope": "all",
        "max_tokens": 400,
        "expect_fragments": ["qwen2.5"],
        "min_citations": 1,
        "description": "Model query should recommend qwen2.5 and warn about qwen3",
    },
    {
        "query": "boot sequence recovery after compaction",
        "scope": "all",
        "max_tokens": 400,
        "expect_fragments": ["phase", "briefing", "personality"],
        "min_citations": 1,
        "description": "Boot query should describe the 5-step phased recovery",
    },
]

# Test collection name — separate from production
TEST_COLLECTION = "test_recall_quality"


class RecallQualityTester:
    """Seeds fixtures, runs queries, checks results."""

    def __init__(self):
        self.storage = StorageLayer()
        self.gateway = MemoryGateway()
        self.passed = 0
        self.failed = 0
        self.results = []

    async def setup(self):
        """Create test collection and seed fixtures."""
        from meridian.config import config as cfg

        # Override collection name for isolation
        self._original_collection = cfg.qdrant_collection
        cfg.qdrant_collection = TEST_COLLECTION

        await self.storage.init_db()

        # Ensure test collection exists (recreate for clean state)
        from qdrant_client.models import Distance, VectorParams
        try:
            self.storage.qdrant.delete_collection(TEST_COLLECTION)
        except Exception:
            pass

        self.storage.qdrant.create_collection(
            collection_name=TEST_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        print(f"Seeding {len(FIXTURES)} fixture memories into '{TEST_COLLECTION}'...")

        # Import workers for embedding
        from meridian.workers import WorkerPool
        workers = WorkerPool()

        for fix in FIXTURES:
            result = await self.storage.remember(
                {
                    "content": fix["content"],
                    "type": fix["type"],
                    "tags": fix["tags"],
                    "importance": fix["importance"],
                    "source": fix.get("source", "test"),
                    "related_to": [],
                },
                workers,
            )
            status = "stored" if result.get("stored") else f"REJECTED: {result.get('reasons', 'unknown')}"
            print(f"  [{fix['id']}] {status}")

        await workers.close()
        print(f"Seeded. Collection has {self.storage.qdrant.get_collection(TEST_COLLECTION).points_count} points.\n")

    async def run_tests(self):
        """Run all test queries and check results."""
        print(f"Running {len(TEST_QUERIES)} recall quality tests...\n")

        for i, test in enumerate(TEST_QUERIES, 1):
            print(f"--- Test {i}/{len(TEST_QUERIES)}: {test['description']} ---")
            print(f"  Query: \"{test['query']}\"")

            result = await self.gateway.recall(
                test["query"],
                test["scope"],
                test["max_tokens"],
                self.storage,
            )

            summary = result["results"]
            sources = result["sources"]
            citation_count = summary.count("[mem_")

            # Check expected fragments
            missing_fragments = []
            for frag in test["expect_fragments"]:
                if frag.lower() not in summary.lower():
                    missing_fragments.append(frag)

            # Check citation count
            citation_ok = citation_count >= test["min_citations"]

            # Determine pass/fail
            passed = len(missing_fragments) == 0 and citation_ok

            if passed:
                self.passed += 1
                print(f"  PASS  | Fragments: all found | Citations: {citation_count} (min: {test['min_citations']})")
            else:
                self.failed += 1
                reasons = []
                if missing_fragments:
                    reasons.append(f"missing fragments: {missing_fragments}")
                if not citation_ok:
                    reasons.append(f"citations: {citation_count} < {test['min_citations']}")
                print(f"  FAIL  | {'; '.join(reasons)}")
                print(f"  Response preview: {summary[:300]}")

            self.results.append({
                "query": test["query"],
                "description": test["description"],
                "passed": passed,
                "citation_count": citation_count,
                "missing_fragments": missing_fragments,
                "response_length": len(summary),
                "source_count": len(sources),
            })
            print()

    async def teardown(self):
        """Clean up test collection and restore config."""
        from meridian.config import config as cfg
        try:
            self.storage.qdrant.delete_collection(TEST_COLLECTION)
            print(f"Cleaned up test collection '{TEST_COLLECTION}'.")
        except Exception as e:
            print(f"Warning: cleanup failed: {e}")

        # Restore original collection name
        if hasattr(self, "_original_collection"):
            cfg.qdrant_collection = self._original_collection

        await self.storage.close()
        await self.gateway.close()

    def report(self):
        """Print summary and return exit code."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"RECALL QUALITY REPORT: {self.passed}/{total} passed")
        print(f"{'='*60}")

        if self.failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - {r['description']}: missing={r['missing_fragments']}, citations={r['citation_count']}")

        # Write JSON report
        report_path = Path(__file__).parent / "recall_quality_report.json"
        report_path.write_text(json.dumps(self.results, indent=2))
        print(f"\nDetailed report: {report_path}")

        return 0 if self.failed == 0 else 1


async def main():
    tester = RecallQualityTester()
    try:
        await tester.setup()
        await tester.run_tests()
    finally:
        await tester.teardown()
    return tester.report()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
