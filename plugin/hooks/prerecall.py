#!/usr/bin/env python3
"""Plugin hook: UserPromptSubmit — automatic memory pre-recall.

Embeds the user prompt, queries Qdrant for similar memories, and injects
top hits as additionalContext. This turns Meridian from "memory you ask for"
into "memory that shows up when relevant."

Designed for the Meridian Claude Code plugin. Requires:
- Ollama running with nomic-embed-text model
- Qdrant running with a 'memories' collection
"""

import json
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "nomic-embed-text"
COLLECTION = "memories"
PROJECT_ID = "default"

SIMILARITY_THRESHOLD = 0.70
MAX_RESULTS = 3
MIN_PROMPT_LENGTH = 10


def http_post(url, payload, timeout=3.0):
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def embed(text):
    result = http_post(f"{OLLAMA_URL}/api/embed",
        {"model": EMBED_MODEL, "input": text, "keep_alive": "10m"}, timeout=3.0)
    if result and "embeddings" in result and result["embeddings"]:
        return result["embeddings"][0]
    return None


def query_qdrant(vector, limit=10):
    payload = {
        "query": vector,
        "filter": {
            "must": [{"key": "project_id", "match": {"value": PROJECT_ID}}],
            "must_not": [{"key": "source", "match": {"value": "episodic_ingest"}}],
        },
        "limit": limit,
        "with_payload": True,
    }
    result = http_post(f"{QDRANT_URL}/collections/{COLLECTION}/points/query",
        payload, timeout=2.0)
    if result and "result" in result and "points" in result["result"]:
        return result["result"]["points"]
    return []


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except Exception:
        hook_input = {}

    prompt = hook_input.get("prompt", "")
    if len(prompt) < MIN_PROMPT_LENGTH:
        sys.exit(0)

    vec = embed(prompt)
    if not vec:
        sys.exit(0)

    hits = query_qdrant(vec, limit=MAX_RESULTS * 2)
    relevant = []
    for hit in hits:
        score = hit.get("score", 0)
        payload = hit.get("payload", {})
        if payload.get("superseded_by"):
            continue
        imp = payload.get("importance", 3)
        # Lower threshold for high-importance memories
        threshold = SIMILARITY_THRESHOLD + {5: -0.07, 4: -0.05}.get(imp, 0)
        if score >= threshold:
            relevant.append(hit)
        if len(relevant) >= MAX_RESULTS:
            break

    if relevant:
        hints = []
        for h in relevant:
            p = h.get("payload", {})
            content = p.get("content", "")[:300]
            hints.append(f"[{p.get('type','')}/imp{p.get('importance',3)} {p.get('mem_id','?')}] {content}")

        result = {
            "additionalContext": "[PRE-RECALL] Relevant memories:\n" + "\n".join(hints)
        }
        print(json.dumps(result))

    sys.exit(0)


if __name__ == "__main__":
    main()
