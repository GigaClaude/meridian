"""Extract facts from NOISE memories via Gateway, then purge originals.

Reads audit_report.json, batches NOISE entries 10 at a time through the
Gateway 14B model to extract any genuinely useful facts, stores them as
clean memories, then deletes the noise from Qdrant.

Usage:
    python scripts/extract_and_purge.py --dry-run    # preview only
    python scripts/extract_and_purge.py               # extract + purge
"""

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from meridian.config import config
from meridian.storage import StorageLayer
from meridian.gateway import MemoryGateway

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("extract_purge")

EXTRACT_PROMPT = """You are a memory curator. Below are {count} raw conversation fragments that were ingested as memories but contain noise (speaker prefixes, conversational filler, etc).

Your job: extract any genuinely NEW, USEFUL facts, decisions, patterns, or warnings from these fragments. Distill them into clean, standalone memory statements.

RULES:
1. Each extracted fact must be a single, self-contained statement with no speaker prefixes
2. Deduplicate across fragments — if 3 entries mention the same thing, output ONE clean version
3. Preserve specifics: ports, paths, commands, rationale, error messages
4. Skip: greetings, reactions, filler ("nice!", "let me check"), questions without answers
5. If a fragment contains NO useful facts, skip it entirely
6. Format: one fact per line, prefixed with "FACT[N]: " where N is importance 1-5:
   - 5: Critical architecture decisions, standing directives, identity/personality
   - 4: Important patterns, technical decisions with rationale, gotchas
   - 3: Useful config, port assignments, tool preferences, hardware details
   - 2: General notes, observations, minor details
   - 1: Barely worth keeping, low-signal
7. If there are ZERO useful facts across all fragments, respond with exactly: EMPTY

FRAGMENTS:
{fragments}

Extract facts now:"""


async def run(dry_run: bool = False):
    report_path = Path(__file__).parent.parent / "audit_report.json"
    if not report_path.exists():
        print("ERROR: Run audit_memories.py first")
        return

    report = json.load(open(report_path))
    noise_entries = report["noise"]
    uncertain_entries = report["uncertain"]
    all_to_process = noise_entries + uncertain_entries

    print(f"Entries to process: {len(all_to_process)} ({len(noise_entries)} noise + {len(uncertain_entries)} uncertain)")

    if dry_run:
        print("\n[DRY RUN] Would process these entries:")
        for e in all_to_process[:10]:
            print(f"  {e['id']}: {e['content_preview'][:80]}...")
        if len(all_to_process) > 10:
            print(f"  ... and {len(all_to_process) - 10} more")
        return

    storage = StorageLayer()
    await storage.init_db()
    gateway = MemoryGateway()

    # Batch into groups of 10
    batches = []
    for i in range(0, len(all_to_process), 10):
        batches.append(all_to_process[i:i+10])

    extracted_facts = []
    ids_to_delete = []

    for batch_num, batch in enumerate(batches):
        print(f"\n--- Batch {batch_num + 1}/{len(batches)} ({len(batch)} entries) ---")

        # Need to get full content for each entry (audit only stored preview)
        fragments_text = ""
        batch_ids = []
        for entry in batch:
            mem_id = entry["id"]
            batch_ids.append(mem_id)

            # Get full content from Qdrant
            import uuid
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, mem_id))
            try:
                points = storage.qdrant.retrieve(
                    collection_name=config.qdrant_collection,
                    ids=[point_uuid],
                    with_payload=True,
                )
                if points:
                    full_content = points[0].payload.get("content", entry["content_preview"])
                else:
                    full_content = entry["content_preview"]
            except Exception:
                full_content = entry["content_preview"]

            fragments_text += f"\n--- [{mem_id}] ---\n{full_content}\n"

        # Send to Gateway for extraction
        prompt = EXTRACT_PROMPT.format(count=len(batch), fragments=fragments_text)
        result = await gateway._generate(prompt, max_tokens=1500)

        if not result or result.startswith("[Gateway error"):
            logger.warning(f"Gateway failed for batch {batch_num + 1}: {result}")
            continue

        # Parse extracted facts
        if result.strip() == "EMPTY":
            print(f"  EMPTY — no useful facts in this batch")
        else:
            batch_facts = []
            for line in result.strip().split("\n"):
                line = line.strip()
                # Parse FACT[N]: format (importance-tagged)
                imp_match = re.match(r"FACT\[(\d)\]:\s*(.*)", line)
                if imp_match:
                    importance = int(imp_match.group(1))
                    fact = imp_match.group(2).strip()
                    if fact:
                        batch_facts.append((fact, importance))
                elif line.startswith("FACT:"):
                    # Fallback for old format without importance
                    fact = line[5:].strip()
                    if fact:
                        batch_facts.append((fact, 2))
                elif line and not line.startswith("EMPTY"):
                    # Sometimes model doesn't use FACT: prefix
                    if len(line) > 20 and not line.startswith("[") and not line.startswith("---"):
                        batch_facts.append((line, 2))

            if batch_facts:
                print(f"  Extracted {len(batch_facts)} facts:")
                for fact, imp in batch_facts:
                    print(f"    → [imp={imp}] {fact[:100]}")
                extracted_facts.extend(batch_facts)

        ids_to_delete.extend(batch_ids)

    # Store extracted facts as clean memories
    print(f"\n{'='*50}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total facts extracted: {len(extracted_facts)}")
    print(f"Entries to purge: {len(ids_to_delete)}")

    stored_count = 0
    for fact, importance in extracted_facts:
        try:
            result = await storage.remember({
                "content": fact,
                "type": "note",
                "tags": ["extracted", "cleanup"],
                "importance": importance,
                "source": "audit_extraction",
            })
            stored_count += 1
        except Exception as e:
            logger.warning(f"Failed to store fact: {e}")

    print(f"Stored {stored_count} clean memories")

    # Purge noise entries
    purged = 0
    failed = 0
    for mem_id in ids_to_delete:
        try:
            await storage.delete_memory(mem_id)
            purged += 1
        except Exception as e:
            logger.warning(f"Failed to delete {mem_id}: {e}")
            failed += 1

    print(f"Purged {purged} noise entries ({failed} failures)")

    # Save extraction log
    log_path = Path(__file__).parent.parent / "extraction_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "extracted_facts": extracted_facts,
            "purged_ids": ids_to_delete,
            "purged_count": purged,
            "stored_count": stored_count,
            "failed_count": failed,
        }, f, indent=2)
    print(f"Log written to: {log_path}")

    await storage.close()
    await gateway.close()


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    asyncio.run(run(dry_run=dry))
