#!/usr/bin/env bash
# Monitors the llama-3.3-70b-instruct-q4 production run and performs the
# full transition to gemma-4-31b-it-q5 when it completes.
#
# Run in a detached screen:
#   screen -dmS transition bash -c "cd /srv/shared/schelling/PancsVriend && bash transition_to_gemma.sh 2>&1 | tee logs/transition.log"

set -euo pipefail
cd /srv/shared/schelling/PancsVriend

LLAMA_LOG="logs/run_llama-3.3-70b-instruct-q4.log"
GEMMA_SERVER_LOG="logs/server_gemma.log"
GEMMA_RUN_LOG="logs/run_gemma-4-31b-it-q5.log"
GEMMA_GGUF="/srv/shared/schelling/PancsVriend/llms/gemma-4-31B-it-Q5_K_M.gguf"
GEMMA_LABEL="gemma-4-31b-it-q5"
SERVER_URL="http://localhost:8080/v1/models"
POLL_INTERVAL=300   # check every 5 minutes

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# 1. Poll until llama run is done
# ---------------------------------------------------------------------------
log "Monitoring $LLAMA_LOG for completion..."
while true; do
    if grep -q "^Pipeline completed\." "$LLAMA_LOG" 2>/dev/null; then
        log "Pipeline completed. detected — proceeding with transition."
        break
    fi
    log "Not done yet. Sleeping ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done

# ---------------------------------------------------------------------------
# 2. Send llama completion email
# ---------------------------------------------------------------------------
log "Sending llama completion email..."
.venv/bin/python notify.py \
    "Schelling run complete: llama-3.3-70b-instruct-q4" \
    "Production run finished. Results in experiments_with_llama_cpp/."

# ---------------------------------------------------------------------------
# 3. Kill llama server
# ---------------------------------------------------------------------------
log "Killing llama_server screen..."
screen -S llama_server -X quit || log "llama_server screen already gone — continuing."

# ---------------------------------------------------------------------------
# 4. Swap configs to gemma
# ---------------------------------------------------------------------------
log "Updating llama_cpp_server.yaml to gemma GGUF..."
sed -i "s|model: \".*\.gguf\"|model: \"$GEMMA_GGUF\"|" configs/llama_cpp_server.yaml

log "Updating llama_cpp_simulation_run.yaml model label..."
sed -i "s|llm_model: \"llama-3.3-70b-instruct-q4\"|llm_model: \"$GEMMA_LABEL\"|" configs/llama_cpp_simulation_run.yaml

# ---------------------------------------------------------------------------
# 5. Start gemma server
# ---------------------------------------------------------------------------
log "Starting gemma server..."
screen -dmS llama_server bash -c \
    "cd /srv/shared/schelling/PancsVriend && .venv/bin/python -m llama_cpp.server --config_file configs/llama_cpp_server.yaml 2>&1 | tee $GEMMA_SERVER_LOG"

# ---------------------------------------------------------------------------
# 6. Wait for server ready (poll up to 30 minutes)
# ---------------------------------------------------------------------------
log "Waiting for gemma server on $SERVER_URL..."
READY=0
for i in $(seq 1 360); do
    if curl -sf "$SERVER_URL" | grep -q '"data"'; then
        log "Server ready after $((i * 5))s."
        READY=1
        break
    fi
    sleep 5
done

if [ "$READY" -eq 0 ]; then
    log "ERROR: Gemma server did not become ready within 30 minutes. Aborting."
    .venv/bin/python notify.py \
        "Schelling ERROR: gemma server failed to start" \
        "The gemma-4-31b-it-q5 server did not respond on port 8080 after 30 minutes. Manual intervention required."
    exit 1
fi

# ---------------------------------------------------------------------------
# 7. Send gemma start email
# ---------------------------------------------------------------------------
log "Sending gemma start email..."
.venv/bin/python notify.py \
    "Schelling run started: $GEMMA_LABEL" \
    "Gemma server is up. Running smoke test before production launch."

# ---------------------------------------------------------------------------
# 8. Smoke test
# ---------------------------------------------------------------------------
log "Running smoke test..."
if .venv/bin/python run_llm_probability_simulation_analysis.py \
        --config-yaml configs/llama_cpp_simulation_run.yaml \
        --config-profile smoke_test; then
    log "Smoke test PASSED."
else
    log "ERROR: Smoke test FAILED. Aborting production launch."
    .venv/bin/python notify.py \
        "Schelling ERROR: gemma smoke test failed" \
        "Smoke test for gemma-4-31b-it-q5 failed. Production run NOT started. Manual intervention required."
    exit 1
fi

# ---------------------------------------------------------------------------
# 9. Launch gemma production
# ---------------------------------------------------------------------------
log "Launching gemma production run..."
screen -dmS llama_run bash -c \
    "cd /srv/shared/schelling/PancsVriend && .venv/bin/python run_llm_probability_simulation_analysis.py --config-yaml configs/llama_cpp_simulation_run.yaml --config-profile production 2>&1 | tee $GEMMA_RUN_LOG"

log "Gemma production run launched. Monitor with: tail -f $GEMMA_RUN_LOG"
.venv/bin/python notify.py \
    "Schelling production started: $GEMMA_LABEL" \
    "Production run launched: 100 runs x 1000 steps x all scenarios. Log: $GEMMA_RUN_LOG"

log "Transition complete. Exiting."
