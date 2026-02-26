#!/bin/bash
# Verify Meridian setup — run this after install to confirm everything works.
set -uo pipefail

PASS=0
FAIL=0
WARN=0

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

pass() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }
warn() { echo "  ! $1"; WARN=$((WARN + 1)); }

echo "=== Meridian Setup Verification ==="
echo ""

# 1. Python
echo "[1/6] Python environment"
if python3 -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
    pass "Python $(python3 --version 2>&1 | cut -d' ' -f2)"
else
    fail "Python 3.11+ required"
fi

if python3 -c "import meridian" 2>/dev/null || python3 -c "
import sys; sys.path.insert(0, '$(cd "$(dirname "$0")/.." && pwd)'); import meridian
" 2>/dev/null; then
    pass "meridian package importable"
else
    fail "Cannot import meridian — run: pip install -e ."
fi

# 2. Qdrant
echo ""
echo "[2/6] Qdrant"
if curl -sf ${QDRANT_URL}/healthz > /dev/null 2>&1; then
    pass "Qdrant responding at ${QDRANT_URL}"
    COLLECTIONS=$(curl -sf ${QDRANT_URL}/collections | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(len(d.get('result',{}).get('collections',[])))
" 2>/dev/null || echo "?")
    pass "Collections: $COLLECTIONS"
else
    fail "Qdrant not running — start with: scripts/start.sh"
fi

# 3. Ollama
echo ""
echo "[3/6] Ollama"
if curl -sf ${OLLAMA_URL}/api/tags > /dev/null 2>&1; then
    pass "Ollama responding at ${OLLAMA_URL}"
else
    fail "Ollama not running — install from https://ollama.com"
fi

# 4. Models
echo ""
echo "[4/6] Required models"
GATEWAY_MODEL="${GATEWAY_MODEL:-qwen2.5-coder:14b}"
EMBED_MODEL="${EMBED_MODEL:-mxbai-embed-large}"

for MODEL in "$GATEWAY_MODEL" "$EMBED_MODEL"; do
    STATUS=$(curl -sf -o /dev/null -w "%{http_code}" \
        -X POST "${OLLAMA_URL}/api/show" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$MODEL\"}" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        pass "$MODEL available"
    else
        fail "$MODEL not found — run: scripts/pull-models.sh"
    fi
done

# 5. Embedding test
echo ""
echo "[5/6] Embedding pipeline"
EMBED_RESULT=$(curl -sf -X POST "${OLLAMA_URL}/api/embed" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$EMBED_MODEL\", \"input\": \"test embedding\"}" 2>/dev/null)

if [ -n "$EMBED_RESULT" ]; then
    DIM=$(echo "$EMBED_RESULT" | python3 -c "
import sys, json; d=json.load(sys.stdin)
emb = d.get('embeddings', [d.get('embedding', [])])[0]
print(len(emb))
" 2>/dev/null || echo "0")
    if [ "$DIM" = "768" ]; then
        pass "Embedding: ${DIM}d vectors"
    else
        fail "Expected 768d embeddings, got ${DIM}d"
    fi
else
    fail "Embedding request failed"
fi

# 6. Gateway synthesis test
echo ""
echo "[6/6] Gateway synthesis"
SYNTH_RESULT=$(curl -sf -X POST "${OLLAMA_URL}/api/chat" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$GATEWAY_MODEL\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say OK\"}],
        \"stream\": false,
        \"options\": {\"num_predict\": 5}
    }" 2>/dev/null)

if [ -n "$SYNTH_RESULT" ]; then
    CONTENT=$(echo "$SYNTH_RESULT" | python3 -c "
import sys, json; d=json.load(sys.stdin)
print(d.get('message',{}).get('content',''))
" 2>/dev/null || echo "")
    if [ -n "$CONTENT" ]; then
        pass "Gateway model responding"
    else
        fail "Gateway model returned empty response (thinking mode?)"
    fi
else
    fail "Gateway model request failed"
fi

# Summary
echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed, $WARN warnings"
echo "================================"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Fix the failures above, then re-run: bash scripts/verify.sh"
    exit 1
else
    echo ""
    echo "All checks passed. Meridian is ready."
    echo "Next: copy .mcp.json to your project directory and start Claude Code."
    exit 0
fi
