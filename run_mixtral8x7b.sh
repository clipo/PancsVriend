#!/bin/bash
# Runner for Mixtral 8x7B Instruct v0.1 Q5_K_M — port 8083
# GGUF is resolved automatically: checks PancsVriend/llms/ first, then ~/llms/.
set -e

cd /srv/shared/schelling/PancsVriend
source .venv/bin/activate

MODEL_LABEL="mixtral-8x7b-q5"
GGUF_FILENAME="mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
RUN_CFG="configs/llama_cpp_run_mixtral8x7b.yaml"
LOG="logs/run_mixtral8x7b.log"
mkdir -p logs

# --- Native llama-server (continuous batching) settings ---------------------
# We launch llama.cpp's native `llama-server` (NOT `python -m llama_cpp.server`,
# which is single-stream). `-np` slots share ONE loaded model and serve requests
# concurrently via continuous batching. Keep run YAML `processes` ≈ $NP.
# The model path is resolved below via find_gguf.sh; configs/llama_cpp_server_mixtral8x7b.yaml
# is now only for the legacy python server.
PORT=8083
NP=8                                  # server slots = max concurrent requests
CTX=16384                             # TOTAL context, split across slots (16384/8 = 2048/slot)
NGL=-1                                # GPU layers (-1 = all; lower if VRAM OOMs)
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"   # override via env if not on PATH

notify() {
    python notify.py "$1" "$2" 2>/dev/null || echo "[notify] email failed: $1"
}

# Resolve GGUF path (project llms/ → ~/llms/)
MODEL_PATH=$(bash find_gguf.sh "$GGUF_FILENAME") || {
    echo "[$(date)] ERROR: $GGUF_FILENAME not found. Download it first:" | tee -a "$LOG"
    echo "  wget -O ~/llms/$GGUF_FILENAME \\" | tee -a "$LOG"
    echo "    https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/$GGUF_FILENAME" | tee -a "$LOG"
    exit 1
}
echo "[$(date)] Using GGUF: $MODEL_PATH" | tee -a "$LOG"

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
    >> "logs/server_mixtral8x7b.log" 2>&1 &
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
        notify "$MODEL_LABEL FAILED to start" "The llama.cpp server on port $PORT did not become ready. Check logs/server_mixtral8x7b.log."
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
    notify "$MODEL_LABEL smoke test FAILED" "Smoke test failed for $MODEL_LABEL. Check $LOG."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Production run
START_TIME=$(date +%s)
notify "$MODEL_LABEL production started" "Production run launched: 100 runs x 1000 steps x all scenarios. Log: $LOG"
echo "[$(date)] Starting production run..." | tee -a "$LOG"
if python run_llm_probability_simulation_analysis.py \
    --config-yaml "$RUN_CFG" \
    --config-profile production >> "$LOG" 2>&1; then
    END_TIME=$(date +%s)
    ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
    echo "[$(date)] Production run COMPLETED in ${ELAPSED} minutes." | tee -a "$LOG"

    git add experiments_with_llama_cpp/ logs/run_mixtral8x7b.log 2>/dev/null || true
    git commit -m "results: $MODEL_LABEL production run complete (${ELAPSED} min)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" >> "$LOG" 2>&1 || true
    git push >> "$LOG" 2>&1 || true

    notify "$MODEL_LABEL production run COMPLETE" \
"Model: $MODEL_LABEL
Duration: ${ELAPSED} minutes
Results: experiments_with_llama_cpp/
Log: $LOG"
else
    echo "[$(date)] Production run FAILED." | tee -a "$LOG"
    notify "$MODEL_LABEL production run FAILED" "Production run failed after smoke test passed. Check $LOG."
fi

kill $SERVER_PID 2>/dev/null
echo "[$(date)] Server stopped." | tee -a "$LOG"
