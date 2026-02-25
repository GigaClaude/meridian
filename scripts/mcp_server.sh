#!/bin/bash
# Wrapper script to launch Meridian MCP server with correct PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
exec python3 -m meridian.mcp_server "$@"
