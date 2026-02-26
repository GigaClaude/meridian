#!/bin/bash
set -e

# ── Wait for Qdrant ──
if [ -n "$QDRANT_URL" ]; then
    echo "Waiting for Qdrant at $QDRANT_URL ..."
    retries=0
    while ! curl -sf "$QDRANT_URL/healthz" > /dev/null 2>&1; do
        retries=$((retries + 1))
        if [ $retries -ge 30 ]; then
            echo "WARNING: Qdrant not reachable after 30 attempts"
            break
        fi
        sleep 2
    done
    echo "Qdrant is ready"
fi

# ── Wait for Ollama + pull models ──
if [ -n "$OLLAMA_URL" ]; then
    echo "Waiting for Ollama at $OLLAMA_URL ..."
    retries=0
    while ! curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; do
        retries=$((retries + 1))
        if [ $retries -ge 60 ]; then
            echo "WARNING: Ollama not reachable after 60 attempts — starting without it"
            echo "Memory storage will work, but recall synthesis and embedding require Ollama"
            unset OLLAMA_URL
            break
        fi
        sleep 5
    done

    if [ -n "$OLLAMA_URL" ]; then
        echo "Ollama is ready"
        echo "Pulling models (first run may take 10-15 minutes)..."
        /app/scripts/pull-models.sh
    fi
fi

# ── Start web server (background) — provides REST API, chat UI, and healthcheck ──
echo "Starting Meridian web server on port ${MERIDIAN_PORT:-7891}..."
python -m meridian.web &
WEB_PID=$!

# Wait for web server to be ready
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${MERIDIAN_PORT:-7891}/api/health" > /dev/null 2>&1; then
        echo "Web server ready"
        break
    fi
    sleep 1
done

echo "Meridian is ready. Web UI at http://localhost:${MERIDIAN_PORT:-7891}"
echo "Connect Claude Code with:"
echo "  claude mcp add meridian -- docker compose exec -T meridian python -m meridian.mcp_server"

# ── Keep container alive — web server is the main process ──
wait $WEB_PID
