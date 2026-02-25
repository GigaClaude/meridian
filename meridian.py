#!/usr/bin/env python3
"""Meridian: Persistent Memory for Claude Code

Starts both the MCP server (for Claude) and the web server (for the browser UI).

Usage:
    python meridian.py              # Start everything
    python meridian.py --mcp-only   # Start only the MCP server
    python meridian.py --web-only   # Start only the web server
"""

import argparse
import subprocess
import sys
import time

import uvicorn

from meridian.config import config


def main():
    parser = argparse.ArgumentParser(description="Meridian: Persistent Memory for Claude Code")
    parser.add_argument("--mcp-only", action="store_true", help="Start only the MCP server (stdio)")
    parser.add_argument("--web-only", action="store_true", help="Start only the web server")
    args = parser.parse_args()

    if args.mcp_only:
        # Run MCP server directly (stdio transport for Claude Code)
        from meridian.mcp_server import mcp
        mcp.run(transport="stdio")
        return

    if args.web_only:
        print(f"Starting Meridian web server on http://0.0.0.0:{config.port}")
        uvicorn.run(
            "meridian.web:app",
            host=config.host,
            port=config.port,
            reload=False,
            log_level="info",
        )
        return

    # Default: start both
    print("=== Meridian ===")
    print(f"Web UI:     http://localhost:{config.port}")
    print(f"MCP server: stdio (configured via .mcp.json)")
    print()
    print("The MCP server runs via stdio when Claude Code spawns it.")
    print("Starting web server now...")
    print()

    uvicorn.run(
        "meridian.web:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
