"""Audit all Qdrant memories — classify as CLEAN, NOISE, or UNCERTAIN.

Tiered classification:
- Instant NOISE: speaker prefixes, multi-turn brackets
- Allowlisted: [migrated...], code blocks
- 2+ signal threshold for everything else (filler, length, conversational patterns)

Output: JSON report + summary stats.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ── Noise detection ──

SPEAKER_PREFIX = re.compile(r'^\[.+?\]:')
MIGRATED_PREFIX = re.compile(r'^\[migrated', re.IGNORECASE)
CODE_INDICATORS = ['```', 'def ', 'import ', 'if __name__', 'class ', 'async def ', 'return ']
FILLER_WORDS = ['lol', 'hmm', 'okay so', 'yeah', 'let me', 'oh wait', 'uh', 'haha', 'nah']
CONVERSATIONAL_PATTERNS = [
    re.compile(r'\b(you|your)\b.*\?', re.IGNORECASE),  # "can you...?" style
    re.compile(r'^(ok|okay|sure|yeah|yep|nope|nah)\b', re.IGNORECASE),  # starts with filler
    re.compile(r'let me (check|look|think|try)', re.IGNORECASE),
]


def classify(content: str, mem_type: str) -> tuple[str, list[str]]:
    """Classify a memory as CLEAN, NOISE, or UNCERTAIN.

    Returns (classification, [reasons]).
    """
    signals = []

    # ── Allowlists (skip noise check entirely) ──
    if MIGRATED_PREFIX.match(content):
        return "CLEAN", ["migrated_prefix_allowlisted"]

    if any(indicator in content for indicator in CODE_INDICATORS):
        # Has code — could still be noisy if it's "[Giga]: let me write some code..."
        if SPEAKER_PREFIX.match(content):
            signals.append("speaker_prefix_with_code")
            # Don't instant-kill, let it accumulate signals
        else:
            return "CLEAN", ["contains_code"]

    # ── Instant NOISE signals ──
    if SPEAKER_PREFIX.match(content) and not any(indicator in content for indicator in CODE_INDICATORS):
        return "NOISE", ["speaker_prefix"]

    # Count bracket pairs that look like multi-turn: [Name]: ... [Name]:
    bracket_speakers = re.findall(r'\[.+?\]:', content)
    if len(bracket_speakers) > 1:
        return "NOISE", ["multi_turn_speakers"]

    # ── Accumulating signals (need 2+) ──
    word_count = len(content.split())

    if word_count < 5:
        signals.append("too_short")

    if word_count > 500:
        signals.append("too_long")

    filler_count = sum(1 for f in FILLER_WORDS if f in content.lower())
    if filler_count >= 2:
        signals.append(f"filler_words({filler_count})")

    for pattern in CONVERSATIONAL_PATTERNS:
        if pattern.search(content):
            signals.append("conversational_pattern")
            break

    # Check for question-only content (single question, no answer)
    if content.strip().endswith('?') and '\n' not in content.strip() and word_count < 30:
        signals.append("standalone_question")

    # Very low information density — mostly short words
    if word_count > 10:
        avg_word_len = sum(len(w) for w in content.split()) / word_count
        if avg_word_len < 3.5:
            signals.append("low_info_density")

    # ── Classify ──
    if len(signals) >= 2:
        return "NOISE", signals
    elif len(signals) == 1:
        return "UNCERTAIN", signals
    else:
        return "CLEAN", []


def run_audit():
    client = QdrantClient(url="http://localhost:6333", timeout=30, check_compatibility=False)

    results = {"CLEAN": [], "NOISE": [], "UNCERTAIN": []}
    total = 0
    offset = None

    pfilter = Filter(must=[
        FieldCondition(key="project_id", match=MatchValue(value="default")),
    ])

    while True:
        batch = client.scroll(
            collection_name="memories",
            scroll_filter=pfilter,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        points, next_offset = batch

        for p in points:
            total += 1
            payload = p.payload
            mem_id = payload.get("mem_id", str(p.id))
            content = payload.get("content", "")
            mem_type = payload.get("type", "note")
            importance = payload.get("importance", 3)
            volatile = payload.get("volatile", False)
            created_at = payload.get("created_at", "")

            classification, reasons = classify(content, mem_type)

            entry = {
                "id": mem_id,
                "classification": classification,
                "reasons": reasons,
                "type": mem_type,
                "importance": importance,
                "volatile": volatile,
                "content_preview": content[:150],
                "content_length": len(content),
                "created_at": created_at,
            }
            results[classification].append(entry)

        if next_offset is None:
            break
        offset = next_offset

    # Summary
    summary = {
        "total": total,
        "clean": len(results["CLEAN"]),
        "noise": len(results["NOISE"]),
        "uncertain": len(results["UNCERTAIN"]),
        "noise_pct": f"{len(results['NOISE'])/total*100:.1f}%" if total > 0 else "0%",
        "timestamp": datetime.utcnow().isoformat(),
    }

    report = {
        "summary": summary,
        "noise": results["NOISE"],
        "uncertain": results["UNCERTAIN"],
        "clean_count": len(results["CLEAN"]),
        # Only include clean IDs, not full entries (they're fine)
        "clean_ids": [e["id"] for e in results["CLEAN"]],
    }

    # Write report
    report_path = Path(__file__).parent.parent / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"MEMORY AUDIT REPORT")
    print(f"{'='*50}")
    print(f"Total memories:  {summary['total']}")
    print(f"  CLEAN:         {summary['clean']}")
    print(f"  NOISE:         {summary['noise']} ({summary['noise_pct']})")
    print(f"  UNCERTAIN:     {summary['uncertain']}")
    print(f"\nReport written to: {report_path}")

    if results["NOISE"]:
        print(f"\n--- Sample NOISE entries ---")
        for entry in results["NOISE"][:5]:
            print(f"\n  [{entry['id']}] reasons={entry['reasons']}")
            print(f"    {entry['content_preview'][:100]}...")

    if results["UNCERTAIN"]:
        print(f"\n--- UNCERTAIN entries ---")
        for entry in results["UNCERTAIN"]:
            print(f"\n  [{entry['id']}] reasons={entry['reasons']}")
            print(f"    {entry['content_preview'][:100]}...")

    return report


if __name__ == "__main__":
    run_audit()
