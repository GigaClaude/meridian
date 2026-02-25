#!/bin/bash
# Idempotent model pull — checks if model exists before downloading.
set -e

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

pull_if_missing() {
    local model="$1"
    if [ -z "$model" ]; then return; fi

    # Check if model already exists
    status=$(curl -sf -o /dev/null -w "%{http_code}" \
        -X POST "$OLLAMA_URL/api/show" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$model\"}" 2>/dev/null || echo "000")

    if [ "$status" = "200" ]; then
        echo "Model $model already available"
        return
    fi

    echo "Pulling $model (this may take a while on first run)..."
    curl -sf -X POST "$OLLAMA_URL/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$model\", \"stream\": false}" > /dev/null
    echo "Model $model ready"
}

pull_if_missing "${GATEWAY_MODEL:-qwen2.5-coder:14b}"
pull_if_missing "${WORKER_MODEL:-qwen2.5-coder:14b}"
pull_if_missing "${EMBED_MODEL:-nomic-embed-text}"
