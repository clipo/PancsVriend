#!/bin/bash
# Line up the A3 production runs, one model after another. Each model runs its
# own llama-server (started + stopped inside run_a3_model.sh), so ports/VRAM
# never collide across the sequence. If one model fails, the queue logs it,
# notifies, and continues to the next (a single bad model does not block the
# rest).
#
# Order:
#   1. Gemma-4-31B     — A3, completions+grammar        (port 8081)
#   2. Llama-3.3-70B   — A3, chat+grammar (--jinja -rea off, port 8082)
#   3. Qwen3.6-27B     — A3, chat+grammar (--jinja -rea off, port 8084)
#
#   ./run_a3_queue.sh    (long-running; launch in the background)
set -u

cd /srv/shared/schelling/PancsVriend
QLOG="logs/run_a3_queue.log"
mkdir -p logs
PYBIN=".venv/bin/python"

ntfy() { ./ntfy.sh "$1" "$2" "${3:-test_tube}" "${4:-default}"; }
# One-line experiment summary derived from the run YAML (never hardcode counts).
describe() { "$PYBIN" notifications.py describe --config "$1" 2>/dev/null || echo "?"; }

# label | run_cfg | port | model_path | extra llama-server args
QUEUE=(
  "gemma-4-31b-a3|configs/llama_cpp_run_gemma31b_a3.yaml|8081|llms/gemma-4-31B-it-Q5_K_M.gguf|"
  "llama-3.3-70b-a3-chat|configs/llama_cpp_run_llama70b_a3_chat.yaml|8082|llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf|--jinja --reasoning off"
  "qwen3.6-27b-a3-chat|configs/llama_cpp_run_qwen36_27b_a3_chat.yaml|8084|llms/Qwen3.6-27B-Q5_K_M.gguf|--jinja --reasoning off"
)

TOTAL=${#QUEUE[@]}
LINEUP=""
for entry in "${QUEUE[@]}"; do
    IFS='|' read -r label cfg _ _ _ <<< "$entry"
    LINEUP+="$label: $(describe "$cfg")"$'\n'
done
echo "[$(date)] === A3 queue starting: $TOTAL models ===" | tee -a "$QLOG"
printf '%s' "$LINEUP" | tee -a "$QLOG"
ntfy "A3 queue STARTED" "$TOTAL models lined up:
$LINEUP" "arrow_forward" "default"

declare -a RESULTS
idx=0
for entry in "${QUEUE[@]}"; do
    idx=$((idx + 1))
    IFS='|' read -r label cfg port mpath extra <<< "$entry"
    echo "[$(date)] --- [$idx/$TOTAL] $label ---" | tee -a "$QLOG"
    ntfy "A3 queue [$idx/$TOTAL]" "Starting $label — $(describe "$cfg")" "hourglass" "low"

    # shellcheck disable=SC2086  # $extra is intentionally word-split into args
    if bash run_a3_model.sh "$label" "$cfg" "$port" "$mpath" $extra >> "$QLOG" 2>&1; then
        echo "[$(date)] [$idx/$TOTAL] $label OK" | tee -a "$QLOG"
        RESULTS+=("OK   $label")
    else
        echo "[$(date)] [$idx/$TOTAL] $label FAILED (continuing)" | tee -a "$QLOG"
        RESULTS+=("FAIL $label")
        ntfy "A3 queue: $label FAILED" "Continuing to next model. Check logs/run_a3_queue.log and the per-model log." "warning" "high"
    fi
done

echo "[$(date)] === A3 queue done ===" | tee -a "$QLOG"
summary=$(printf '%s\n' "${RESULTS[@]}")
echo "$summary" | tee -a "$QLOG"
ntfy "A3 queue FINISHED" "All $TOTAL models processed:
$summary" "checkered_flag" "high"
