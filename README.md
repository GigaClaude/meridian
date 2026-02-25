# Meridian

**Claude Code forgets everything on compaction. Meridian fixes that.**

Persistent memory with local LLM synthesis — runs on a single consumer GPU, no cloud dependencies. While other memory solutions dump raw search results into context, Meridian has a local Gateway model that synthesizes memories before delivery: 5,000 tokens of search results become 500 tokens of actionable summary. Your context stays clean.

## Why Not Just Use CLAUDE.md?

CLAUDE.md works until you have 50 decisions, 30 patterns, and 20 warnings. Then you're loading 80% of your context window before work even begins. Every session starts fat and gets fatter.

Meridian loads ~1,500 tokens at boot (a YAML briefing with current task, recent decisions, active warnings). Everything else is on-demand — ask for what you need, when you need it. Memories you don't query don't cost tokens.

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

Everything runs locally. No API keys needed for the memory layer. Your data stays on your machine.

## Quick Start

### 1. Install

```bash
git clone https://github.com/GigaClaude/meridian.git
cd meridian
pip install -e .
```

### 2. Pull models

```bash
ollama pull nomic-embed-text       # embeddings (768-dim, ~270MB)
ollama pull qwen2.5-coder:14b     # gateway + workers (~9GB)
```

### 3. Start services

```bash
# Qdrant (vector DB — single binary, no Docker needed)
qdrant --config-path ~/.meridian/qdrant_config.yaml &

# Ollama (if not already running)
ollama serve &
```

### 4. Connect to Claude Code

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

### 5. Add to CLAUDE.md

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

## Web UI

Meridian includes a browser chat UI and REST API:

```bash
python meridian.py --web-only
# Open http://localhost:7891
```

REST endpoints: `/api/memory/recall`, `/api/memory/remember`, `/api/memory/briefing`, `/api/bridge/*` (inter-agent messaging).

## Notes

- Gateway/worker models are configurable. Any Ollama-compatible model works. Avoid qwen3 series (thinking mode consumes output tokens silently).
- Qdrant runs as a single binary — no Docker or container needed.
- The MCP server communicates via stdio. Claude Code manages the lifecycle automatically.

## License

MIT
