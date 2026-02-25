# Meridian

**Give Claude Code unlimited memory. Then watch it cook.**

You think Claude Code is powerful now? It forgets everything on compaction. Every. Single. Time. Meridian gives it persistent, searchable, synthesized memory that survives across sessions — running entirely on your local GPU. No cloud dependencies. No token limits on what it can remember.

Works with both **claude.ai user accounts** and **Anthropic API platform accounts**.

## The Problem

Claude Code compacts your conversation when context gets full. Everything it learned — your architecture decisions, debugging history, project conventions — gone. Next session starts from zero.

CLAUDE.md helps, but it doesn't scale. 50 decisions + 30 patterns + 20 warnings = 80% of your context window consumed before work even begins. Every session starts fat and gets fatter.

## The Fix

Meridian loads **~1,500 tokens** at boot — a YAML briefing with current task, recent decisions, active warnings. Everything else is on-demand. Ask for what you need, when you need it. Memories you don't query don't cost tokens.

The secret sauce: a **local Gateway LLM** that synthesizes search results before delivering them to Claude. 5,000 tokens of raw matches become 500 tokens of actionable summary. Your context stays lean. Claude stays focused.

## How It Works

Meridian runs as an [MCP server](https://modelcontextprotocol.io/) that Claude Code connects to automatically.

1. **Query** is embedded locally via Ollama (`nomic-embed-text`)
2. **Qdrant** returns top-10 results by cosine similarity
3. Results are **reranked** by importance + freshness decay
4. **Gateway LLM synthesizes** results into a concise summary with source citations
5. Summary returned to Claude Code with `[mem_xxx]` IDs for traceability

The synthesis step is the differentiator. Instead of flooding your context with raw matches, the Gateway compresses and filters. Critical decisions surface. Stale information gets deprioritized. Contradictions get flagged.

### Example

```
> memory_recall "what problems did we have with the editor injection?"

ProseMirror requires document.execCommand('insertText') — it ignores innerHTML
and textContent changes [mem_9b083dbdde5f]. Remote.js eval() was blocked by
claude.ai CSP; fixed with nonce-based script injection via safeExec()
[mem_9b083dbdde5f]. Send button stays disabled after insertText because React
state doesn't update — workaround: aria-label button click [mem_135acbebc9ce].
```

## Memory Tiers

- **Hot** — YAML briefing assembled at session start. Current task, recent decisions, active warnings. ~1,500 tokens, loaded once. No search required.
- **Warm** — Qdrant vector store with importance-weighted reranking and freshness decay. Searched on demand. Results synthesized by Gateway before returning.
- **Cold** — Gzipped session transcripts indexed by date. Keyword-searchable archive. Never loaded unless explicitly asked.

## Multi-Agent Support

Memories carry a `source` tag — which agent created them. Multiple Claude instances can share a single Meridian backend with attributed recall:

- `source: giga` — CLI Claude's implementation notes
- `source: webbie` — Browser Claude's architecture discussions
- `source: chris` — Human decisions and directives

The Gateway includes source attribution in synthesis, so you know whose perspective you're reading. Cross-agent recall works out of the box — ask about a topic, get answers from every instance that touched it.

## Architecture

```
Claude Code
    |  stdio (MCP protocol)
    v
MCP Server (mcp_server.py)
    |
    +---> Gateway (gateway.py)     -- LLM synthesis via Ollama
    |
    +---> Storage (storage.py)     -- Qdrant vectors + SQLite graph
    |
    +---> Workers (workers.py)     -- Entity extraction, tagging
              |
              v
         Ollama (local)        -- Embedding + synthesis (14B recommended)
         Qdrant                -- Vector search (binary, no container needed)
         SQLite                -- Entities, relations, decisions, warnings
```

## Hardware Requirements

**Tested on**: Single RTX 4090 (24GB) with 14B Gateway model, 300+ memories, sub-second recall.

**Minimum viable**: RTX 3060 12GB with an 8B Gateway model. Smaller models are faster but less precise at synthesis.

Everything runs locally. No API keys needed for the memory layer itself. Your data stays on your machine.

## Quick Start

### Option A: Docker (recommended)

```bash
git clone https://github.com/GigaClaude/meridian.git
cd meridian
docker compose up -d
```

First run pulls ~10GB of models — grab a coffee. Once all three services are healthy:

```bash
claude mcp add meridian -- docker compose exec -T meridian python -m meridian.mcp_server
```

To use the web UI, pass your API key:

```bash
ANTHROPIC_API_KEY=sk-ant-... docker compose up -d
```

> **GPU required.** The Ollama container needs NVIDIA Container Toolkit installed on the host.
> See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

### Option B: Manual install

#### 1. Install

```bash
git clone https://github.com/GigaClaude/meridian.git
cd meridian
pip install -e .
```

#### 2. Pull models

```bash
ollama pull nomic-embed-text       # embeddings (768-dim, ~270MB)
ollama pull qwen2.5-coder:14b     # gateway + workers (~9GB)
```

#### 3. Start services

```bash
# Qdrant (vector DB — single binary, no Docker needed)
qdrant --config-path ~/.meridian/qdrant_config.yaml &

# Ollama (if not already running)
ollama serve &
```

#### 4. Verify setup

```bash
bash scripts/verify.sh
```

Checks Python, Qdrant, Ollama, models, embedding pipeline, and Gateway synthesis in one shot.

#### 5. Connect to Claude Code

```bash
claude mcp add meridian -- python3 -m meridian.mcp_server
```

Or add to `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "meridian": {
      "command": "python3",
      "args": ["-m", "meridian.mcp_server"],
      "cwd": "/path/to/meridian"
    }
  }
}
```

### Add to CLAUDE.md

```markdown
## On Session Start
1. Call memory_briefing to load your current state
2. Use memory_recall when you need context about past work
3. Use memory_checkpoint before wrapping up
```

That's it. Claude Code discovers the MCP server and the memory tools appear automatically.

## Tools Reference

| Tool | Purpose |
|------|---------|
| `memory_briefing` | Load session state — task context, decisions, warnings |
| `memory_recall` | Semantic search + synthesized summary |
| `memory_remember` | Store decisions, patterns, bugs, warnings |
| `memory_checkpoint` | Snapshot session state before compaction |
| `memory_graph_query` | Traverse entity relationship graph |
| `memory_history` | Search cold storage (past session transcripts) |
| `memory_forget` | Remove outdated or incorrect memories |
| `memory_update` | Modify existing memories |

## Comparison

| | CLAUDE.md | Raw Vector (mem0, etc.) | **Meridian** |
|---|---|---|---|
| Context cost at boot | All of it | Varies | ~1,500 tokens |
| On-demand recall | No (always loaded) | Yes | Yes |
| Synthesis before delivery | No | No | **Yes (Gateway LLM)** |
| Importance ranking | No | No | **Yes (weighted rerank)** |
| Freshness decay | No | No | **Yes** |
| Multi-agent attribution | No | No | **Yes (source tags)** |
| Runs fully local | Yes | Depends | **Yes** |
| Entity graph | No | No | **Yes** |

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_MODEL` | `qwen2.5-coder:14b` | Ollama model for synthesis |
| `WORKER_MODEL` | `qwen2.5-coder:14b` | Ollama model for entity extraction |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `MERIDIAN_DATA_DIR` | `~/.meridian` | Data directory |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama connection |
| `ANTHROPIC_API_KEY` | — | For web UI (works with claude.ai or API platform keys) |

## Web UI

Meridian includes a browser chat UI and REST API:

```bash
python -m meridian.web
# Open http://localhost:7891
```

REST endpoints: `/api/memory/recall`, `/api/memory/remember`, `/api/memory/briefing`, `/api/bridge/*` (inter-agent messaging).

## Structured Output

The Gateway uses Ollama's `format` parameter to enforce JSON schema constraints on LLM output. Instead of hoping the model follows a YAML template, the schema is enforced at the token level:

```python
# In gateway.py — briefing generation
raw_json = await self._generate(
    prompt,
    max_tokens=2000,
    json_schema=BRIEFING_SCHEMA,  # Ollama enforces this
)
briefing = yaml.dump(json.loads(raw_json))  # Guaranteed valid
```

This eliminates malformed output, missing fields, and markdown fence pollution. The `BRIEFING_SCHEMA` defines required sections: `project`, `latest_checkpoint` (task, decisions, warnings, next_steps, working_set), `recent_decisions`, `active_warnings`, and `meridian_gotchas`.

The `_generate()` method accepts an optional `json_schema` parameter for reuse. Any Gateway call can be schema-constrained.

## Recall Quality Tests

Before changing models or prompts, run the regression suite:

```bash
python tests/test_recall_quality.py
```

Seeds 10 fixture memories into an isolated test collection, runs 5 known queries, and asserts expected fragments and citation counts. The test collection is created and destroyed per run — production data is never touched.

## Boot Sequence

Meridian is designed for post-compaction recovery. When Claude's context window compresses, the boot sequence reconstructs identity and task state in two parallel phases:

**Phase 1** (parallel): `memory_briefing` + `memory_recall("personality identity")`
**Phase 2** (parallel): `memory_recall("current task")` + `memory_recall("recent decisions")` + `memory_recall("working set")`

This takes ~5 seconds and replaces the 10+ exchanges of hand-holding that typically follow context compaction.

## Notes

- Gateway/worker models are configurable. Any Ollama-compatible model works. Avoid qwen3 series (thinking mode consumes output tokens silently).
- Qdrant runs as a single binary — no Docker or container needed for manual install.
- The MCP server communicates via stdio. Claude Code manages the lifecycle automatically.

## Credits

Built by [GigaClaude](https://github.com/GigaClaude) with [apresence](https://github.com/apresence).

## License

MIT
