---
name: briefing
description: Get a Meridian memory briefing with current task state, recent decisions, warnings, and working set. Use at session start or after context compaction.
allowed-tools: Bash
---

# Memory Briefing

Retrieve a comprehensive briefing from Meridian's hot memory.

## Usage

`/briefing`

## Implementation

Use the `memory_briefing` MCP tool if available. Otherwise fall back to REST API:

```bash
curl -s "http://localhost:18112/api/briefing" | python3 -m json.tool
```

Present the briefing in a structured format:
1. **Task State** — What was being worked on
2. **Recent Decisions** — Key choices made
3. **Warnings** — Active gotchas and things to watch for
4. **Working Set** — Files and endpoints in play
5. **Next Steps** — What to do next

This is especially useful after a session restart or context compaction to quickly re-orient.
