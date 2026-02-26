---
name: recall
description: Search Meridian memory for past decisions, patterns, debugging history, or stored context. Use when you need information that was saved in a previous session.
allowed-tools: Bash
argument-hint: <search query>
---

# Memory Recall

Search Meridian's persistent memory for relevant context.

## Usage

`/recall <what you're looking for>`

Examples:
- `/recall architecture decisions for the API`
- `/recall why we chose PostgreSQL over MongoDB`
- `/recall debugging the auth token refresh bug`

## Implementation

Use the `memory_recall` MCP tool if available:

```
memory_recall(query="$ARGUMENTS", scope="all", max_tokens=800)
```

If MCP tools are unavailable, fall back to the REST API:

```bash
curl -s -X POST "http://localhost:18112/api/recall" \
  -H "Content-Type: application/json" \
  -d '{"query": "$ARGUMENTS", "scope": "all", "max_tokens": 800}'
```

Present the synthesized results to the user. Include source memory IDs for traceability.
