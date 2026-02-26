---
name: remember
description: Store important information in Meridian memory. Use for decisions, patterns, debugging insights, or anything that should persist across sessions.
allowed-tools: Bash
argument-hint: <what to remember>
---

# Store Memory

Save important context to Meridian's persistent memory.

## Usage

`/remember <what to save>`

Examples:
- `/remember We chose JWT over session cookies because the API is stateless`
- `/remember Port 8080 is reserved for the dev proxy, 8443 for production`
- `/remember The flaky test in auth.test.ts is caused by race condition in token refresh`

## Implementation

Analyze the content to determine the best memory type:
- **decision**: Choices made with rationale ("chose X because Y")
- **pattern**: Recurring approaches or conventions
- **debug**: Bug findings and fixes
- **warning**: Gotchas, things to avoid
- **note**: General reference information

Assess importance (1-5):
- 1-2: Minor details, temporary info
- 3: Standard useful info (default)
- 4: Important decisions, key architecture
- 5: Critical rules, standing orders

Use `memory_remember` MCP tool if available. Otherwise fall back to REST API:

```bash
curl -s -X POST "http://localhost:18112/api/remember" \
  -H "Content-Type: application/json" \
  -d '{"content": "$ARGUMENTS", "type": "<detected_type>", "importance": <1-5>}'
```

Confirm what was stored and the assigned memory ID.
