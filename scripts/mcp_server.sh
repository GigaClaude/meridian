#!/bin/bash
# Wrapper script to launch Meridian MCP server with correct PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Agent identity — scopes checkpoints and briefings to this agent
export MERIDIAN_AGENT_ID="${MERIDIAN_AGENT_ID:-giga}"

exec python3 -m meridian.mcp_server "$@"
