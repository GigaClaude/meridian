"""Fix importance levels on extracted memories and prune stale ones.

The extract_and_purge.py script hardcoded importance=2 for all extracted facts.
This script reviews them and either:
  - Deletes stale/completed/trivial entries
  - Bumps importance for genuinely valuable facts

Usage:
    python scripts/fix_extracted_importance.py --dry-run   # preview
    python scripts/fix_extracted_importance.py              # apply
"""

import sys
import re
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointVectors

# Patterns that indicate a memory is stale (task completed or no longer relevant)
STALE_PATTERNS = [
    # Completed TODO items
    r"(?i)no freshness.decay factor exists yet",
    r"(?i)implement decay.freshness scoring",
    r"(?i)update.*recall.*method.*identity routing.*reranking",
    r"(?i)personality\.md is not in meridian",
    r"(?i)qdrant is not running and needs",
    r"(?i)qdrant needs to be set up",
    r"(?i)needs to be verified for clean startup",
    r"(?i)mcp server is missing the.*cwd.*field",
    r"(?i)anthropic.api.key is not set",
    r"(?i)add a tag filter to memory_recall",
    r"(?i)use.*memory_remember.*to import personality",
    r"(?i)tighten the gateway synthesis prompt",
    r"(?i)update readme requirements section",
    r"(?i)import.*personality.*into meridian",
    r"(?i)add.*matchany import",
    r"(?i)update.*recall method to use identity routing",
    # Transient/session-specific noise
    r"task notification received.*task ID",
    r"task status killed.*summary",
    r"command produced zero bytes",
    r"claude is hallucinating instead",
    r"chris mentions booting something",
    r"chris has identified a bug and is fixing",
    r"chris performed brain surgery",
    r"another instance of claude may already be running",
    r"hot memory might be stale or thin",
    r"mcp server killed and needs session restart",
    r"(?i)mcp server.*PID \d+.*running as a stdio",
    r"(?i)checkpoint.*chk_\w+.*saved",
    r"(?i)giga.*restarting the web server",
    r"(?i)giga investigates why.*claude -p",
    r"(?i)chris is type-less",
    r"(?i)overengineered hardware requirements",
    r"(?i)instruct claude to print raw data",
    r"(?i)debug with.*--debug",
    r"(?i)quick eyeball.*imported memory",
    r"(?i)giga needs.*MatchAny",
    r"(?i)giga confirms.*memory is locked in",
    r"(?i)midnight checkpoint has the current state",
    # Superseded/duplicated info
    r"(?i)ranking behavior is as expected",
    r"(?i)fresh imp-3 memory can barely edge out",
    r"(?i)smashed check for tags.*minimum length",
    r"(?i)memory recall works across separate cli",
    r"(?i)claude successfully used.*meridian.*memory_recall",
    r"(?i)mcp server is registered but failing",
    r"(?i)claude can see.*\.mcp\.json.*but not load",
    r"(?i)try installing the mcp server globally",
    r"(?i)globally registered mcp allows persistent",
    r"(?i)mcp server initializes and responds",
    r"(?i)headless.*-p.*mode might not read",
    r"(?i)the web server has health.status endpoints but no memory api",
    r"(?i)web server has fastapi.*storage.*gateway wired up.*rest endpoints.*need",
    r"(?i)giga is building the fastapi host",
    r"(?i)33 chunks with reasonable classification",
    r"(?i)all 33 memories are tagged with.*migrated",
    r"(?i)qdrant is running and the embedding model",
    r"(?i)giga wired up memory access but timing",
    r"(?i)api key issue resolved",
]

# Keywords that indicate high-value content (bump to importance 3-4)
HIGH_VALUE_PATTERNS = {
    4: [
        r"(?i)architecture.*three.tier|three tiers",
        r"(?i)licensing strategy",
        r"(?i)boot sequence.*5.step|phased recovery",
        r"(?i)priority 5 directives",
        r"(?i)quality gate.*tiered noise",
        r"(?i)identity.*routing.*reranking.*implemented",
        r"(?i)decay.*formula.*freshness",
        r"(?i)structured output.*format.*json schema",
    ],
    3: [
        r"(?i)port.*assignments|port.*6333.*11434",
        r"(?i)cuda_visible_devices",
        r"(?i)gpu.*4090|gpu.*8000",
        r"(?i)qwen.*thinking mode|qwen3.*thinking",
        r"(?i)fastmcp.*standalone|mcp.*fastmcp",
        r"(?i)oauth2.*gmail|gmail.*oauth",
        r"(?i)backup.*7z.*compression",
        r"(?i)executor.*websocket.*bridge",
        r"(?i)cors.*wide open",
        r"(?i)self.directing directive.*critical priority",
    ],
}


def should_delete(content: str) -> bool:
    for pattern in STALE_PATTERNS:
        if re.search(pattern, content):
            return True
    return False


def get_new_importance(content: str) -> int | None:
    for importance, patterns in sorted(HIGH_VALUE_PATTERNS.items(), reverse=True):
        for pattern in patterns:
            if re.search(pattern, content):
                return importance
    return None


def main(dry_run: bool = False):
    client = QdrantClient(host="localhost", port=6333, check_compatibility=False)

    # Get all extracted memories
    result = client.scroll(
        collection_name="memories",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="source", match=MatchValue(value="audit_extraction")
                ),
            ]
        ),
        limit=300,
        with_payload=True,
    )

    points = result[0]
    print(f"Found {len(points)} extracted memories\n")

    to_delete = []
    to_bump = []
    unchanged = 0

    for pt in points:
        content = pt.payload.get("content", "")

        if should_delete(content):
            to_delete.append((pt.id, content[:100]))
        else:
            new_imp = get_new_importance(content)
            if new_imp:
                to_bump.append((pt.id, new_imp, content[:100]))
            else:
                unchanged += 1

    print(f"DELETE:    {len(to_delete)}")
    print(f"BUMP:     {len(to_bump)}")
    print(f"KEEP AS-IS: {unchanged}")
    print()

    if to_delete:
        print("=== DELETIONS ===")
        for pid, preview in to_delete:
            print(f"  DEL {pid[:12]}... | {preview}")
        print()

    if to_bump:
        print("=== IMPORTANCE BUMPS ===")
        for pid, new_imp, preview in to_bump:
            print(f"  IMP={new_imp} {pid[:12]}... | {preview}")
        print()

    if dry_run:
        print("[DRY RUN] No changes applied.")
        return

    # Apply deletions
    deleted = 0
    for pid, _ in to_delete:
        try:
            client.delete(
                collection_name="memories",
                points_selector=[pid],
            )
            deleted += 1
        except Exception as e:
            print(f"  Failed to delete {pid}: {e}")

    # Apply importance bumps
    bumped = 0
    for pid, new_imp, _ in to_bump:
        try:
            client.set_payload(
                collection_name="memories",
                payload={"importance": new_imp},
                points=[pid],
            )
            bumped += 1
        except Exception as e:
            print(f"  Failed to bump {pid}: {e}")

    print(f"\nApplied: {deleted} deletions, {bumped} importance bumps")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    main(dry_run=dry)
