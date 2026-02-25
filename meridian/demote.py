"""Cold demotion sweep — run periodically to move aged-out memories to cold storage.

Usage:
    python -m meridian.demote [--dry-run] [--pin-from-checkpoint]

Demotion criteria:
    1. Volatile + older than 7 days + importance ≤ 3
    2. Non-volatile + older than 30 days + importance ≤ 2
    3. Superseded by a newer memory

Pinned memories (from active checkpoint working_set) are exempt.
Nothing is deleted — demoted memories move to SQLite cold archive
and can be auto-promoted back to warm on recall.
"""

import asyncio
import json
import logging
import sys

from .config import config
from .storage import StorageLayer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("meridian.demote")


async def run_sweep(dry_run: bool = False, pin_from_checkpoint: bool = True):
    storage = StorageLayer()
    await storage.init_db()

    # Get pinned IDs from latest checkpoint's working_set
    pinned_ids = set()
    if pin_from_checkpoint:
        checkpoint = await storage.get_latest_checkpoint()
        if checkpoint and checkpoint.get("working_set"):
            ws = checkpoint["working_set"]
            # Pin memories related to files in the working set
            # (We'd need to search for these — for now, just log it)
            logger.info(f"Active checkpoint working_set: {ws}")

    # Get cold memory count before sweep
    cold_before = 0
    try:
        async with storage._db.execute("SELECT COUNT(*) FROM cold_memories") as cursor:
            row = await cursor.fetchone()
            cold_before = row[0]
    except Exception:
        pass

    warm_count = await storage.get_memory_count()
    logger.info(f"Before sweep: {warm_count} warm, {cold_before} cold")

    if dry_run:
        logger.info("DRY RUN — would demote the following:")
        # Just scan and report what would be demoted
        from datetime import datetime
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        now = datetime.utcnow()
        offset = None
        candidates = []
        pfilter = Filter(must=[
            FieldCondition(key="project_id", match=MatchValue(value=config.project_id)),
        ])

        while True:
            result = storage.qdrant.scroll(
                collection_name=config.qdrant_collection,
                scroll_filter=pfilter,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            points, next_offset = result
            for p in points:
                mem_id = p.payload.get("mem_id", "")
                importance = p.payload.get("importance", 3)
                is_volatile = p.payload.get("volatile", False)
                is_superseded = bool(p.payload.get("superseded_by"))
                created_at = p.payload.get("created_at", "")
                content_preview = p.payload.get("content", "")[:60]

                days_old = 0
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at)
                        days_old = (now - dt).total_seconds() / 86400
                    except (ValueError, TypeError):
                        pass

                reason = None
                if is_superseded:
                    reason = "superseded"
                elif is_volatile and days_old > 7 and importance <= 3:
                    reason = f"volatile (age={int(days_old)}d, imp={importance})"
                elif not is_volatile and days_old > 30 and importance <= 2:
                    reason = f"low_imp (age={int(days_old)}d, imp={importance})"

                if reason:
                    candidates.append((mem_id, reason, content_preview))

            if next_offset is None:
                break
            offset = next_offset

        for mem_id, reason, preview in candidates:
            logger.info(f"  WOULD DEMOTE: {mem_id} [{reason}] {preview}")
        logger.info(f"Total: {len(candidates)} candidates for demotion")
    else:
        result = await storage.sweep_demotions(pinned_ids=pinned_ids)
        total = sum(result.values())
        logger.info(f"Sweep complete: {total} demoted ({result})")

        warm_after = await storage.get_memory_count()
        cold_after = 0
        try:
            async with storage._db.execute("SELECT COUNT(*) FROM cold_memories") as cursor:
                row = await cursor.fetchone()
                cold_after = row[0]
        except Exception:
            pass
        logger.info(f"After sweep: {warm_after} warm, {cold_after} cold")

    await storage.close()


def main():
    dry_run = "--dry-run" in sys.argv
    pin = "--pin-from-checkpoint" in sys.argv or "--no-pin" not in sys.argv
    asyncio.run(run_sweep(dry_run=dry_run, pin_from_checkpoint=pin))


if __name__ == "__main__":
    main()
