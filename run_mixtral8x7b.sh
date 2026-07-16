#!/bin/bash
# Runner for Mixtral 8x7B Instruct v0.1 Q5_K_M — port 8083
# GGUF is resolved automatically: checks PancsVriend/llms/ first, then ~/llms/.
set -e

: "${GMAIL_APP_PW:?Error: set GMAIL_APP_PW before running this script}"
export GMAIL_APP_PW

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
# concurrently via continuous batching. The slot count is NOT hardcoded here: it is
# `processes` from the run YAML (production profile), read below into $NP/$CTX. Pin
# that value from a slot_sweep/ throughput benchmark, not a memory-fit guess.
# The model path is resolved below via find_gguf.sh; configs/llama_cpp_server_mixtral8x7b.yaml
# is now only for the legacy python server.
PORT=8083
NGL=-1                                # GPU layers (-1 = all; lower if VRAM OOMs)
CTX_PER_SLOT=2048                     # usable context window each slot gets
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

# Slot count comes straight from `processes` in the run YAML (production profile):
# the server's -np is kept equal to the client concurrency from ONE value. No probe,
# no auto-cap -- pin the number from a slot_sweep/ throughput benchmark
# (slot_sweep/sweep_slots.py). llama.cpp -c is the TOTAL context split across slots,
# so -c = per-slot context * slots.
NP="$(python - "$RUN_CFG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
def procs(block):
    ca = (block or {}).get("contexts_args", block) if isinstance(block, dict) else None
    return ca.get("processes") if isinstance(ca, dict) else None
p = procs((cfg.get("profiles") or {}).get("production"))
if p is None:
    p = procs(cfg)                       # inherit top-level default
if p is None or str(p).lower() == "auto":
    sys.exit(1)                          # no explicit integer to launch with
print(int(p))
PY
)"
if ! [[ "$NP" =~ ^[0-9]+$ ]]; then
    echo "[$(date)] ERROR: no explicit integer \`processes\` in $RUN_CFG (production profile); pin one from a slot_sweep benchmark." | tee -a "$LOG"
    exit 1
fi
CTX=$(( NP * CTX_PER_SLOT ))
echo "[$(date)] Slots from $RUN_CFG processes: -np $NP  -c $CTX" | tee -a "$LOG"

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
