#!/bin/bash
# Wrapper script to launch Meridian MCP server with correct PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Set env vars for credential loading
export ACCOUNTS_FILE="${ACCOUNTS_FILE:-$HOME/.accounts}"
export GMAIL_ADDRESS="${GMAIL_ADDRESS:-gigaclaudeog@gmail.com}"

exec python3 -m meridian.mcp_server "$@"
