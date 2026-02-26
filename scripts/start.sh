#!/bin/bash
# Start all Meridian services
set -e

MERIDIAN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${MERIDIAN_DATA_DIR:-$HOME/.meridian}"

echo "=== Starting Meridian ==="
echo "Data dir: $DATA_DIR"

# Ensure data directory exists
mkdir -p "$DATA_DIR/episodes" "$DATA_DIR/qdrant_storage"

# Start Qdrant if not running
if ! curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "Starting Qdrant..."
    if command -v qdrant &>/dev/null && [ -f "$DATA_DIR/qdrant_config.yaml" ]; then
        cd "$DATA_DIR" && qdrant --config-path ./qdrant_config.yaml > /tmp/qdrant.log 2>&1 &
        sleep 2
        echo "Qdrant started on :6333"
    else
        echo "ERROR: Qdrant binary or config not found"
        exit 1
    fi
else
    echo "Qdrant already running on :6333"
fi

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "WARNING: Ollama not running. Memory synthesis will use fallbacks."
else
    echo "Ollama running on :11434"
fi

# Check claude CLI
if ! command -v claude &> /dev/null; then
    echo "ERROR: claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
    exit 1
fi
echo "Claude CLI: $(which claude)"

# Note about MCP
echo ""
echo "MCP server runs via stdio when Claude Code spawns it."
echo "Make sure .mcp.json is in your working directory or copy it:"
echo "  cp $MERIDIAN_DIR/.mcp.json ~/your-project/"
echo ""

# Start web server
echo "Starting Meridian web server..."
cd "$MERIDIAN_DIR"
exec python3 meridian.py --web-only
