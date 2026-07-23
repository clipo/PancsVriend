#!/bin/bash
# Generic A3-style production runner for one model, mirroring the completed
# Llama-3.3-70B A3 run: start a native llama-server, smoke-test, then run the
# full A3 production suite (20 runs x 100 steps x 6 scenarios), commit results,
# with ntfy phone notifications throughout.
#
# Usage:
#   run_a3_model.sh <label> <run_cfg> <port> <model_path> [extra llama-server args...]
#
# Examples:
#   run_a3_model.sh gemma-4-31b-a3 configs/llama_cpp_run_gemma31b_a3.yaml \
#       8081 llms/gemma-4-31B-it-Q5_K_M.gguf
#   run_a3_model.sh llama-3.3-70b-a3-chat configs/llama_cpp_run_llama70b_a3_chat.yaml \
#       8082 llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf --jinja --reasoning off
#
# Notifications (identical to the Llama-3.3 A3 run): coarse stage pings via
# ./ntfy.sh, plus fine-grained per-scenario / per-run ETA pings from
# ./watch_progress.sh tailing the run log. Subscribe to topic
# "schelling-runs-7f3a9c" (or set NTFY_TOPIC) in the ntfy app. No email/secret
# needed — everything goes to ntfy.
set -u

cd /srv/shared/schelling/PancsVriend
source .venv/bin/activate 2>/dev/null || true
# `python` is not reliably on PATH in non-interactive shells even after activate;
# call the venv interpreter directly (project convention).
PYBIN=".venv/bin/python"

MODEL_LABEL="${1:?usage: run_a3_model.sh <label> <run_cfg> <port> <model_path> [server args...]}"
RUN_CFG="${2:?missing run_cfg}"
PORT="${3:?missing port}"
MODEL_PATH="${4:?missing model_path}"
shift 4
EXTRA_SERVER_ARGS=("$@")   # e.g. --jinja --reasoning off (chat endpoints)

SLUG="$(printf '%s' "$MODEL_LABEL" | tr -c 'A-Za-z0-9._-' '_')"
LOG="logs/run_${SLUG}.log"
SERVER_LOG="logs/server_${SLUG}.log"
mkdir -p logs

NGL=-1                                # GPU layers (-1 = all; lower if VRAM OOMs)
CTX_PER_SLOT=2048                     # usable context window each slot gets
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/srv/shared/schelling/llama.cpp/build/bin/llama-server}"

ntfy() { ./ntfy.sh "$1" "$2" "${3:-test_tube}" "${4:-default}"; }

SERVER_PID=""
WATCH_PID=""
cleanup() {
    [ -n "$WATCH_PID" ]  && kill "$WATCH_PID"  2>/dev/null
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

echo "[$(date)] === $MODEL_LABEL | cfg=$RUN_CFG port=$PORT ===" | tee -a "$LOG"

# Pre-flight: native binary present?
if ! "$LLAMA_SERVER_BIN" --version 2>&1 | grep -qi version; then
    echo "[$(date)] ERROR: native llama-server not found at '$LLAMA_SERVER_BIN'." | tee -a "$LOG"
    ntfy "$MODEL_LABEL launch FAILED" "llama-server binary not found at '$LLAMA_SERVER_BIN'." "rotating_light" "high"
    exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "[$(date)] ERROR: GGUF not found: $MODEL_PATH" | tee -a "$LOG"
    ntfy "$MODEL_LABEL launch FAILED" "GGUF not found: $MODEL_PATH" "rotating_light" "high"
    exit 1
fi

# Slot count from `processes` in the run YAML (production profile).
# llama.cpp -c is TOTAL context split across slots: -c = per-slot context * slots.
NP="$("$PYBIN" - "$RUN_CFG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
def procs(block):
    ca = (block or {}).get("contexts_args", block) if isinstance(block, dict) else None
    return ca.get("processes") if isinstance(ca, dict) else None
p = procs((cfg.get("profiles") or {}).get("production"))
if p is None:
    p = procs(cfg)
if p is None or str(p).lower() == "auto":
    sys.exit(1)
print(int(p))
PY
)"
if ! [[ "$NP" =~ ^[0-9]+$ ]]; then
    echo "[$(date)] ERROR: no explicit integer \`processes\` in $RUN_CFG (production profile)." | tee -a "$LOG"
    ntfy "$MODEL_LABEL launch FAILED" "No explicit \`processes\` in $RUN_CFG." "rotating_light" "high"
    exit 1
fi
CTX=$(( NP * CTX_PER_SLOT ))
echo "[$(date)] Slots: -np $NP  -c $CTX  extra=[${EXTRA_SERVER_ARGS[*]:-}]" | tee -a "$LOG"

# Human-readable run description for notifications (accurate for any profile —
# A3 20-run or A2 5-run — instead of hardcoded text).
META="$("$PYBIN" - "$RUN_CFG" <<'PY'
import sys, yaml, os
cfg = yaml.safe_load(open(sys.argv[1]))
ca = cfg["profiles"]["production"]["contexts_args"]
scen = os.path.splitext(os.path.basename(ca.get("scenario_file") or "context_scenarios.py"))[0]
print(f'{ca.get("runs")} {ca.get("max_steps")} {ca.get("llm_style") or "legacy"} {scen}')
PY
)"
read -r RUNS STEPS STYLE SCEN <<< "$META"
DESC="$RUNS runs x $STEPS steps, all scenarios ($SCEN, $STYLE)"

echo "[$(date)] Starting server (native, -np $NP -c $CTX) on port $PORT..." | tee -a "$LOG"
"$LLAMA_SERVER_BIN" \
    -m "$MODEL_PATH" --alias "$MODEL_LABEL" \
    -ngl "$NGL" -fa on \
    -np "$NP" -c "$CTX" \
    --host 127.0.0.1 --port "$PORT" \
    "${EXTRA_SERVER_ARGS[@]}" \
    >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[$(date)] Server PID $SERVER_PID" | tee -a "$LOG"

# Wait for server to be ready (up to 10 min — large GGUFs load slowly)
echo "[$(date)] Waiting for server on port $PORT..." | tee -a "$LOG"
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "[$(date)] Server ready." | tee -a "$LOG"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date)] Server process died during startup. Check $SERVER_LOG." | tee -a "$LOG"
        ntfy "$MODEL_LABEL FAILED to start" "Server process exited during load. Check $SERVER_LOG (arch unsupported? OOM?)." "rotating_light" "high"
        exit 1
    fi
    sleep 5
    if [ "$i" -eq 120 ]; then
        echo "[$(date)] Server failed to become ready after 10 minutes." | tee -a "$LOG"
        ntfy "$MODEL_LABEL FAILED to start" "Server on port $PORT not ready in 10 min. Check $SERVER_LOG." "rotating_light" "high"
        exit 1
    fi
done

ntfy "$MODEL_LABEL launching" "Server up on port $PORT (-np $NP). A3 prompt. Running smoke test next." "rocket" "default"

# Smoke test (minimal: 2 runs x 2 steps)
echo "[$(date)] Running smoke test..." | tee -a "$LOG"
if "$PYBIN" run_llm_probability_simulation_analysis.py \
    --config-yaml "$RUN_CFG" \
    --config-profile smoke_test >> "$LOG" 2>&1; then
    echo "[$(date)] Smoke test PASSED." | tee -a "$LOG"
else
    echo "[$(date)] Smoke test FAILED." | tee -a "$LOG"
    ntfy "$MODEL_LABEL smoke test FAILED" "Smoke test failed. Check $LOG." "rotating_light" "high"
    exit 1
fi

# Start the log->ntfy watcher now (after smoke, before production) so per-run
# and per-scenario pings track the real run without smoke-test noise.
./watch_progress.sh "$LOG" &
WATCH_PID=$!
echo "[$(date)] watch_progress.sh PID $WATCH_PID tailing $LOG" | tee -a "$LOG"

# Production run
START_TIME=$(date +%s)
ntfy "$MODEL_LABEL production STARTED" "20 runs x 100 steps x 6 scenarios (A3). Per-scenario pings follow." "checkered_flag" "default"
echo "[$(date)] Starting production run..." | tee -a "$LOG"
if "$PYBIN" run_llm_probability_simulation_analysis.py \
    --config-yaml "$RUN_CFG" \
    --config-profile production >> "$LOG" 2>&1; then
    END_TIME=$(date +%s)
    ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
    echo "[$(date)] Production run COMPLETED in ${ELAPSED} minutes." | tee -a "$LOG"

    echo "[$(date)] Committing results to git..." | tee -a "$LOG"
    git add experiments_with_llama_cpp/ configs/ run_a3_model.sh run_a3_queue.sh "$LOG" 2>/dev/null || true
    git commit -m "results: $MODEL_LABEL A3 production run complete (${ELAPSED} min)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" >> "$LOG" 2>&1 || true
    git push >> "$LOG" 2>&1 || true
    echo "[$(date)] Git push done." | tee -a "$LOG"

    ntfy "$MODEL_LABEL production COMPLETE" \
"Model: $MODEL_LABEL
Duration: ${ELAPSED} minutes
Results: experiments_with_llama_cpp/
Git pushed to master." "white_check_mark" "high"
else
    echo "[$(date)] Production run FAILED." | tee -a "$LOG"
    ntfy "$MODEL_LABEL production FAILED" "Production run failed after smoke passed. Check $LOG." "rotating_light" "high"
    exit 1
fi

echo "[$(date)] Done: $MODEL_LABEL." | tee -a "$LOG"
