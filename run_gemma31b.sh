#!/bin/bash
# Runner for Gemma 4 31B Q5_K_M — port 8081
set -e

cd /srv/shared/schelling/PancsVriend
source .venv/bin/activate

MODEL_LABEL="gemma-4-31b-q5"
RUN_CFG="configs/llama_cpp_run_gemma31b.yaml"
LOG="logs/run_gemma31b.log"
mkdir -p logs

# --- Native llama-server (continuous batching) settings ---------------------
# We launch llama.cpp's native `llama-server` (NOT `python -m llama_cpp.server`,
# which is single-stream). `-np` slots share ONE loaded model and serve requests
# concurrently via continuous batching. Keep run YAML `processes` ≈ $NP.
# configs/llama_cpp_server_gemma31b.yaml is now only for the legacy python server.
PORT=8081
MODEL_PATH="/srv/shared/schelling/PancsVriend/llms/gemma-4-31B-it-Q5_K_M.gguf"
NP=8                                  # server slots = max concurrent requests
CTX=16384                             # TOTAL context, split across slots (16384/8 = 2048/slot)
NGL=-1                                # GPU layers (-1 = all; lower if VRAM OOMs)
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"   # override via env if not on PATH

notify() {
    python notify.py "$1" "$2" 2>/dev/null || echo "[notify] email failed: $1"
}

# Pre-flight: the native binary may not ship with the pip llama-cpp-python wheel.
if ! "$LLAMA_SERVER_BIN" --version 2>&1 | grep -qi version; then
    echo "[$(date)] ERROR: native llama-server not found at '$LLAMA_SERVER_BIN'." | tee -a "$LOG"
    notify "$MODEL_LABEL launch FAILED" "Native llama-server binary not found at '$LLAMA_SERVER_BIN'. Install it (prebuilt CUDA release / conda-forge llama.cpp / build from source) or set the LLAMA_SERVER_BIN env var to its path."
    exit 1
fi

echo "[$(date)] Starting $MODEL_LABEL server (native, -np $NP -c $CTX) on port $PORT..." | tee -a "$LOG"
"$LLAMA_SERVER_BIN" \
    -m "$MODEL_PATH" --alias "$MODEL_LABEL" \
    -ngl "$NGL" -fa on \
    -np "$NP" -c "$CTX" \
    --host 127.0.0.1 --port "$PORT" \
    >> "logs/server_gemma31b.log" 2>&1 &
SERVER_PID=$!
echo "[$(date)] Server PID $SERVER_PID" | tee -a "$LOG"

# Wait for server to be ready
echo "[$(date)] Waiting for server on port $PORT..." | tee -a "$LOG"
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "[$(date)] Server ready." | tee -a "$LOG"
        break
    fi
    sleep 5
    if [ $i -eq 120 ]; then
        echo "[$(date)] Server failed to start after 10 minutes." | tee -a "$LOG"
        notify "$MODEL_LABEL FAILED to start" "The llama.cpp server on port $PORT did not become ready within 10 minutes. Check logs/server_gemma31b.log."
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done

# Smoke test
echo "[$(date)] Running smoke test..." | tee -a "$LOG"
if python run_llm_probability_simulation_analysis.py \
    --config-yaml "$RUN_CFG" \
    --config-profile smoke_test >> "$LOG" 2>&1; then
    echo "[$(date)] Smoke test PASSED." | tee -a "$LOG"
else
    echo "[$(date)] Smoke test FAILED." | tee -a "$LOG"
    notify "$MODEL_LABEL smoke test FAILED" "Smoke test failed for $MODEL_LABEL. Check logs/run_gemma31b.log for details."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Production run
START_TIME=$(date +%s)
echo "[$(date)] Starting production run..." | tee -a "$LOG"
if python run_llm_probability_simulation_analysis.py \
    --config-yaml "$RUN_CFG" \
    --config-profile production >> "$LOG" 2>&1; then
    END_TIME=$(date +%s)
    ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
    echo "[$(date)] Production run COMPLETED in ${ELAPSED} minutes." | tee -a "$LOG"

    # Git commit and push
    echo "[$(date)] Committing results to git..." | tee -a "$LOG"
    git add experiments_with_llama_cpp/ logs/run_gemma31b.log 2>/dev/null || true
    git commit -m "results: $MODEL_LABEL production run complete (${ELAPSED} min)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" >> "$LOG" 2>&1 || true
    git push >> "$LOG" 2>&1 || true
    echo "[$(date)] Git push done." | tee -a "$LOG"

    notify "$MODEL_LABEL production run COMPLETE" \
"Model: $MODEL_LABEL
Duration: ${ELAPSED} minutes
Results: experiments_with_llama_cpp/
Log: $LOG

Git results have been pushed to master."
else
    echo "[$(date)] Production run FAILED." | tee -a "$LOG"
    notify "$MODEL_LABEL production run FAILED" "Production run failed for $MODEL_LABEL after smoke test passed. Check logs/run_gemma31b.log."
fi

kill $SERVER_PID 2>/dev/null
echo "[$(date)] Server stopped." | tee -a "$LOG"
