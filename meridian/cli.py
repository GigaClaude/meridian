#!/usr/bin/env python3
"""meridian — CLI for wetware access to the Meridian memory system.

Unix-philosophy tool: one command, does one thing, composes with pipes.
Wraps the Meridian web API (localhost:7891) so Chris (or any human)
can search, write, and read memories from the terminal.

EXP-011: Wetware Meridian Access. Architecture by Webbie.

Usage:
    meridian search "what did we decide about the gateway model?"
    meridian remember "Always use keep_alive for Ollama calls" --type decision --tags ollama,performance
    meridian remember "The 4090 can't truly parallelize same-model inference" --type pattern
    meridian briefing
    meridian log
    echo "some insight" | meridian remember --type note
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.parse

API_BASE = os.environ.get("MERIDIAN_URL", "http://localhost:18101")


def _post(endpoint: str, data: dict) -> dict:
    """POST JSON to Meridian API, return parsed response."""
    payload = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{API_BASE}{endpoint}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"Error: Can't reach Meridian at {API_BASE} — is the server running?", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)


def _get(endpoint: str, params: dict | None = None, timeout: int = 30) -> dict:
    """GET from Meridian API, return parsed response."""
    url = f"{API_BASE}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"Error: Can't reach Meridian at {API_BASE} — is the server running?", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)


def cmd_search(args):
    """Search memories by semantic similarity."""
    query = " ".join(args.query)
    if not query:
        print("Usage: meridian search <query>", file=sys.stderr)
        sys.exit(1)

    result = _post("/api/memory/recall", {
        "query": query,
        "scope": args.scope,
        "max_tokens": args.max_tokens,
    })

    # The recall endpoint returns synthesized text via the gateway LLM
    text = result.get("results", result.get("result", "No results."))
    print(text)

    # Show sources if available
    sources = result.get("sources", [])
    if sources:
        print(f"\n--- Sources ({len(sources)}) ---")
        for s in sources:
            sid = s.get("id", "?")
            stype = s.get("type", "?")
            rel = s.get("relevance", 0)
            print(f"  [{sid}] {stype} (relevance: {rel:.2f})")


def cmd_remember(args):
    """Store a new memory."""
    content = " ".join(args.content) if args.content else None

    # Read from stdin if no content provided
    if not content:
        if sys.stdin.isatty():
            print("Usage: meridian remember <content>", file=sys.stderr)
            print("  or:  echo <content> | meridian remember --type note", file=sys.stderr)
            sys.exit(1)
        content = sys.stdin.read().strip()

    if not content:
        print("Error: empty content", file=sys.stderr)
        sys.exit(1)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    result = _post("/api/memory/remember", {
        "content": content,
        "type": args.type,
        "tags": tags,
        "importance": args.importance,
        "source": "wetware",
    })

    mem_id = result.get("id", "unknown")
    stored = result.get("stored", False)
    if stored:
        print(f"Stored: {mem_id}")
        print(f"  Type: {args.type}, Importance: {args.importance}, Source: wetware")
        if tags:
            print(f"  Tags: {', '.join(tags)}")
    else:
        print(f"Warning: store returned {json.dumps(result)}", file=sys.stderr)


def cmd_briefing(args):
    """Get the current memory briefing."""
    result = _get("/api/memory/briefing", {"project_id": args.project}, timeout=120)
    briefing = result.get("briefing", "No briefing available.")
    print(briefing)


def cmd_log(args):
    """Show recent channel messages (who said what)."""
    result = _get("/api/channel/history", {"limit": args.limit})
    messages = result.get("messages", [])

    if not messages:
        print("No recent messages.")
        return

    from datetime import datetime
    for msg in messages:
        sender = msg.get("sender", "?")
        content = msg.get("content", "")
        ts = msg.get("ts", 0)
        time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "??:??:??"
        # Truncate long messages for log view
        preview = content[:120] + ("..." if len(content) > 120 else "")
        print(f"[{time_str}] {sender}: {preview}")


def cmd_health(args):
    """Check Meridian service health."""
    result = _get("/api/health")
    status = result.get("status", "unknown")
    print(f"Status: {status}")
    for key in ("qdrant", "ollama", "claude_cli"):
        val = result.get(key, "?")
        print(f"  {key}: {val}")


def main():
    parser = argparse.ArgumentParser(
        prog="meridian",
        description="CLI for wetware access to the Meridian memory system",
    )
    sub = parser.add_subparsers(dest="command")

    # search
    p_search = sub.add_parser("search", aliases=["s", "recall"], help="Search memories")
    p_search.add_argument("query", nargs="+", help="Natural language search query")
    p_search.add_argument("--scope", default="all",
                          choices=["all", "decisions", "patterns", "debug", "entities", "warnings"],
                          help="Filter by memory type")
    p_search.add_argument("--max-tokens", type=int, default=800, help="Max response tokens")
    p_search.set_defaults(func=cmd_search)

    # remember
    p_remember = sub.add_parser("remember", aliases=["r", "store"], help="Store a memory")
    p_remember.add_argument("content", nargs="*", help="What to remember (or pipe via stdin)")
    p_remember.add_argument("--type", "-t", default="note",
                            choices=["decision", "pattern", "debug", "entity", "warning", "note"],
                            help="Memory type (default: note)")
    p_remember.add_argument("--tags", default="", help="Comma-separated tags")
    p_remember.add_argument("--importance", "-i", type=int, default=3,
                            choices=[1, 2, 3, 4, 5], help="Importance 1-5 (default: 3)")
    p_remember.set_defaults(func=cmd_remember)

    # briefing
    p_briefing = sub.add_parser("briefing", aliases=["b"], help="Get memory briefing")
    p_briefing.add_argument("--project", default="default", help="Project ID")
    p_briefing.set_defaults(func=cmd_briefing)

    # log
    p_log = sub.add_parser("log", aliases=["l"], help="Show recent channel messages")
    p_log.add_argument("--limit", "-n", type=int, default=20, help="Number of messages")
    p_log.set_defaults(func=cmd_log)

    # health
    p_health = sub.add_parser("health", aliases=["h"], help="Check service health")
    p_health.set_defaults(func=cmd_health)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
