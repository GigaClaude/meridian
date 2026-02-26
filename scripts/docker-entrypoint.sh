#!/bin/bash
set -e

# Wait for Ollama if URL is set (GPU profile or external host)
if [ -n "$OLLAMA_URL" ]; then
    echo "Checking Ollama at $OLLAMA_URL ..."
    retries=0
    max_retries=60
    while ! curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; do
        retries=$((retries + 1))
        if [ $retries -ge $max_retries ]; then
            echo "WARNING: Ollama not reachable after ${max_retries} attempts — starting without it"
            echo "Memory storage will work, but recall synthesis and embedding require Ollama"
            unset OLLAMA_URL
            break
        fi
        sleep 5
    done

    if [ -n "$OLLAMA_URL" ]; then
        echo "Ollama is ready"
        /app/scripts/pull-models.sh
    fi
fi

exec python -m meridian.mcp_server
