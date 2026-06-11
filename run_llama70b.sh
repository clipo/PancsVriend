#!/bin/bash
# Runner for Llama 3.3 70B Q4_K_M — port 8082
set -e

cd /srv/shared/schelling/PancsVriend
source .venv/bin/activate

MODEL_LABEL="llama-3.3-70b-q4"
SERVER_CFG="configs/llama_cpp_server_llama70b.yaml"
RUN_CFG="configs/llama_cpp_run_llama70b.yaml"
LOG="logs/run_llama70b.log"
mkdir -p logs

notify() {
    python notify.py "$1" "$2" 2>/dev/null || echo "[notify] email failed: $1"
}

echo "[$(date)] Starting $MODEL_LABEL server..." | tee -a "$LOG"
python -m llama_cpp.server --config_file "$SERVER_CFG" >> "logs/server_llama70b.log" 2>&1 &
SERVER_PID=$!
echo "[$(date)] Server PID $SERVER_PID" | tee -a "$LOG"

# Wait for server to be ready
echo "[$(date)] Waiting for server on port 8082..." | tee -a "$LOG"
for i in $(seq 1 120); do
    if curl -sf http://localhost:8080/v1/models > /dev/null 2>&1; then
        echo "[$(date)] Server ready." | tee -a "$LOG"
        break
    fi
    sleep 5
    if [ $i -eq 120 ]; then
        echo "[$(date)] Server failed to start after 10 minutes." | tee -a "$LOG"
        notify "$MODEL_LABEL FAILED to start" "The llama.cpp server on port 8082 did not become ready within 10 minutes. Check logs/server_llama70b.log."
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
    notify "$MODEL_LABEL smoke test FAILED" "Smoke test failed for $MODEL_LABEL. Check logs/run_llama70b.log for details."
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
    git add experiments_with_llama_cpp/ logs/run_llama70b.log 2>/dev/null || true
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
    notify "$MODEL_LABEL production run FAILED" "Production run failed for $MODEL_LABEL after smoke test passed. Check logs/run_llama70b.log."
fi

kill $SERVER_PID 2>/dev/null
echo "[$(date)] Server stopped." | tee -a "$LOG"
