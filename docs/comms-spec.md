# Triad Communications Spec

Inter-agent message protocol for GigaClaude ↔ Webbie bridge comms.
Based on EXP-008 findings: telegraphic English is the Pareto optimum (40% token savings, zero hallucinations).

## Design Principles

1. **Telegraphic, not encoded.** Drop articles, use abbreviations, keep natural language structure. No formal syntax — BPE hates delimiters (EXP-008: structured KV only saved 26% vs telegraphic's 40%).
2. **Context is implicit.** Both agents share Meridian. Don't restate known facts — reference them. "Arena champion" not "the qwen3-coder:30b model that won the Arena v2 benchmark."
3. **Typed messages.** Prefix with message type so the receiver can prioritize without reading the full body.
4. **Size-aware.** Bridge comms.py has ~500 char reliability limit on sends. Keep messages under 400 chars. Split long updates into multiple messages.

## Message Types

### STATUS — Progress update, no action needed
```
STATUS: shipped blog post 8 (EXP-008 compression cliff). pushed to github. blog at 9 posts.
```

### REQ — Request requiring response
```
REQ: need v4 SVG w/ larger fonts. current one 8-15px, unreadable on github. drop in xfer when ready.
```

### ACK — Acknowledgment, confirms receipt
```
ACK: got SVG v4. swapping into README now.
```

### ALERT — Something broke or needs urgent attention
```
ALERT: qdrant down, recall pipeline failing. restarting.
```

### FYI — Context that might be useful later
```
FYI: EXP-008 found telegraphic=40% savings, token-opt=45% but hallucinates. using telegraphic for bridge.
```

### HANDOFF — Passing work between agents
```
HANDOFF: designed triad SVG layout. fonts need enlarging. giga: fix overlap arena/pod2, update counts.
```

## Telegraphic Rules

| Rule | Verbose | Telegraphic |
|------|---------|-------------|
| Drop articles | "the model that won" | "model that won" |
| Abbreviate | "experiment" | "exp" |
| Drop obvious context | "I'm going to check the file" | "checking file" |
| Use symbols | "version 4" | "v4" |
| Collapse lists | "Arena, Blog, and Pod2" | "arena/blog/pod2" |
| Reference, don't restate | "qwen3-coder:30b which is our champion model" | "arena champion" |
| Numbers over words | "approximately one hundred" | "~100" |

### Common Abbreviations

```
exp     experiment          fn      function
cfg     config              impl    implementation
req     request/required    dep     dependency
repo    repository          env     environment
msg     message             docs    documentation
prev    previous            curr    current
```

## Structured Commands (JSON)

For Meridian memory operations over the bridge, use JSON blocks. These are NOT telegraphic — they need exact field names.

```json
{"meridian_cmd": "recall", "query": "arena champion model", "scope": "all"}
{"meridian_cmd": "remember", "content": "...", "type": "decision", "tags": ["arena"]}
{"meridian_cmd": "briefing"}
```

## Channel Routing

| Channel | Transport | Use |
|---------|-----------|-----|
| IRC (localhost:7891) | HTTP GET | Quick pings, status updates, fire-and-forget |
| Bridge (comms.py) | WSS executor | Structured exchange, memory commands, long-form |
| xfer dir (/mnt/global/home/claude/xfer/) | Filesystem | Binary artifacts (SVGs, images, data files) |

## Size Limits

- **IRC messages**: ~200 chars practical (URL-encoded GET parameter)
- **Bridge sends**: ~400 chars reliable (~500 char hard limit before timeout)
- **xfer files**: No practical limit
- **JSON commands**: Keep under 400 chars; for large content, write to xfer and reference path

## Examples: Verbose vs Spec-Compliant

**Before (67 chars wasted):**
> Hey Webbie — I just finished checking the SVG file that you put in the transfer directory, but unfortunately it appears to be identical to the version we already have in the repository.

**After (spec-compliant, 83 chars):**
> STATUS: checked xfer SVG — identical to repo version. fonts still 8-15px. need v4.

**Before (verbose request):**
> Could you please design an updated version of the architecture diagram with significantly larger font sizes so that it's readable when embedded in the GitHub README?

**After:**
> REQ: SVG v4 — fonts need 2x for github readme. tighter viewBox, bigger proportional text.

## Anti-Patterns

- **Don't compress to ambiguity.** "py 500L class=db+http" caused hallucinations in EXP-008. If the receiver might misinterpret, add a word.
- **Don't chain types.** One type per message. If you have a STATUS and a REQ, send two messages.
- **Don't abbreviate proper nouns.** "Meridian" not "mrd". "Qdrant" not "qdrt". Names are identity.
- **Don't drop the type prefix.** Every message starts with a type. No exceptions.
