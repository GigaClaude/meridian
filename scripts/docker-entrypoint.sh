#!/bin/bash
set -e

# Pull models if Ollama is reachable
if [ -n "$OLLAMA_URL" ]; then
    /app/scripts/pull-models.sh
fi

exec python -m meridian.mcp_server
