#!/usr/bin/env python3
"""Migrate existing CLAUDE.md / MEMORY.md files into Meridian's memory store.

Usage:
    python scripts/migrate_markdown.py [path_to_markdown_files...]

If no paths given, searches common locations:
    - ./CLAUDE.md, ./MEMORY.md
    - ~/.claude/CLAUDE.md
    - ./.claude/projects/*/memory/*.md
"""

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from meridian.storage import StorageLayer
from meridian.schemas import MemoryRecord


def parse_markdown_sections(text: str) -> list[dict]:
    """Split markdown into sections by headers."""
    sections = []
    current_header = "General"
    current_content = []

    for line in text.split("\n"):
        if line.startswith("#"):
            if current_content:
                sections.append({
                    "header": current_header,
                    "content": "\n".join(current_content).strip(),
                })
            current_header = line.lstrip("#").strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append({
            "header": current_header,
            "content": "\n".join(current_content).strip(),
        })

    return [s for s in sections if s["content"]]


def classify_section(header: str, content: str) -> tuple[str, int]:
    """Classify a section into a memory type and importance."""
    header_lower = header.lower()
    content_lower = content.lower()

    if any(w in header_lower for w in ["warning", "caution", "don't", "gotcha", "bug"]):
        return "warning", 4
    if any(w in header_lower for w in ["decision", "chose", "architecture", "design"]):
        return "decision", 4
    if any(w in header_lower for w in ["pattern", "convention", "style", "rule"]):
        return "pattern", 3
    if any(w in header_lower for w in ["debug", "fix", "error", "issue"]):
        return "debug", 3
    if any(w in content_lower for w in ["don't", "never", "warning", "careful", "breaks"]):
        return "warning", 4
    if any(w in content_lower for w in ["decided", "chose", "switched", "using"]):
        return "decision", 3
    return "note", 2


def extract_tags(header: str, content: str) -> list[str]:
    """Extract tags from section content."""
    tags = []
    # Use header words as tags
    for word in re.findall(r'\b[a-zA-Z][a-zA-Z0-9_.-]+\b', header):
        if len(word) > 2 and word.lower() not in ("the", "and", "for", "not"):
            tags.append(word.lower())
    # Look for code/tech terms in content
    for match in re.findall(r'`([^`]+)`', content):
        if len(match) < 30:
            tags.append(match.lower())
    return list(set(tags))[:8]


async def migrate_file(filepath: Path, storage: StorageLayer):
    """Migrate a single markdown file into Meridian."""
    print(f"\nMigrating: {filepath}")
    text = filepath.read_text()
    sections = parse_markdown_sections(text)

    count = 0
    for section in sections:
        if len(section["content"]) < 20:
            continue

        mem_type, importance = classify_section(section["header"], section["content"])
        tags = extract_tags(section["header"], section["content"])
        tags.append(f"migrated:{filepath.name}")

        content = f"[{section['header']}] {section['content']}"
        if len(content) > 2000:
            content = content[:2000] + "..."

        record = MemoryRecord(
            content=content,
            type=mem_type,
            tags=tags,
            importance=importance,
            project_id=storage.project_id,
        )
        await storage.store_memory(record)
        count += 1
        print(f"  [{mem_type}] {section['header'][:50]}... ({len(content)} chars)")

    print(f"  Migrated {count} sections from {filepath.name}")
    return count


async def main():
    paths = []

    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:] if Path(p).exists()]
    else:
        # Search common locations
        candidates = [
            Path("CLAUDE.md"),
            Path("MEMORY.md"),
            Path.home() / ".claude" / "CLAUDE.md",
        ]
        # Also search project memory dirs
        claude_projects = Path.home() / ".claude" / "projects"
        if claude_projects.exists():
            for project_dir in claude_projects.iterdir():
                memory_dir = project_dir / "memory"
                if memory_dir.exists():
                    candidates.extend(memory_dir.glob("*.md"))

        paths = [p for p in candidates if p.exists()]

    if not paths:
        print("No markdown files found to migrate.")
        print("Usage: python scripts/migrate_markdown.py [files...]")
        return

    print(f"Found {len(paths)} files to migrate:")
    for p in paths:
        print(f"  - {p}")

    storage = StorageLayer()
    await storage.init_db()

    total = 0
    for path in paths:
        total += await migrate_file(path, storage)

    await storage.close()
    print(f"\nMigration complete: {total} memories imported from {len(paths)} files.")


if __name__ == "__main__":
    asyncio.run(main())
