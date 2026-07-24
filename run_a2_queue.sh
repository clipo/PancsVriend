#!/bin/bash
# A2 endpoint-comparison batch: each of the 3 models run on BOTH endpoints
# (completions+grammar and chat+grammar), A2 prompt style (scenarios_a2.py).
# Reuses the generic run_a3_model.sh runner (config-driven; not A3-specific).
# Each model+endpoint starts and stops its own llama-server, so ports/VRAM
# never collide. On failure the queue logs, notifies, and continues.
#
# Order (6):
#   1. gemma-4-31b     A2 completions+grammar        (8081)
#   2. gemma-4-31b     A2 chat+grammar (--jinja -rea off, 8081)
#   3. llama-3.3-70b   A2 completions+grammar        (8082)
#   4. llama-3.3-70b   A2 chat+grammar (--jinja -rea off, 8082)
#   5. qwen3.6-27b     A2 completions+grammar        (8084)
#   6. qwen3.6-27b     A2 chat+grammar (--jinja -rea off, 8084)
#
#   ./run_a2_queue.sh    (long-running; launch in the background)
set -u

cd /srv/shared/schelling/PancsVriend
QLOG="logs/run_a2_queue.log"
mkdir -p logs
PYBIN=".venv/bin/python"

ntfy() { ./ntfy.sh "$1" "$2" "${3:-test_tube}" "${4:-default}"; }
# One-line experiment summary derived from the run YAML (never hardcode counts).
describe() { "$PYBIN" notifications.py describe --config "$1" 2>/dev/null || echo "?"; }

# label | run_cfg | port | model_path | extra llama-server args
QUEUE=(
  "gemma-4-31b-a2|configs/llama_cpp_run_gemma31b_a2_completions.yaml|8081|llms/gemma-4-31B-it-Q5_K_M.gguf|"
  "gemma-4-31b-a2-chat|configs/llama_cpp_run_gemma31b_a2_chat.yaml|8081|llms/gemma-4-31B-it-Q5_K_M.gguf|--jinja --reasoning off"
  "llama-3.3-70b-a2|configs/llama_cpp_run_llama70b_a2_completions.yaml|8082|llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf|"
  "llama-3.3-70b-a2-chat|configs/llama_cpp_run_llama70b_a2_chat.yaml|8082|llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf|--jinja --reasoning off"
  "qwen3.6-27b-a2|configs/llama_cpp_run_qwen36_27b_a2_completions.yaml|8084|llms/Qwen3.6-27B-Q5_K_M.gguf|"
  "qwen3.6-27b-a2-chat|configs/llama_cpp_run_qwen36_27b_a2_chat.yaml|8084|llms/Qwen3.6-27B-Q5_K_M.gguf|--jinja --reasoning off"
)

TOTAL=${#QUEUE[@]}
LINEUP=""
for entry in "${QUEUE[@]}"; do
    IFS='|' read -r label cfg _ _ _ <<< "$entry"
    LINEUP+="$label: $(describe "$cfg")"$'\n'
done
echo "[$(date)] === A2 queue starting: $TOTAL runs ===" | tee -a "$QLOG"
printf '%s' "$LINEUP" | tee -a "$QLOG"
ntfy "A2 queue STARTED" "$TOTAL runs lined up:
$LINEUP" "arrow_forward" "default"

declare -a RESULTS
idx=0
for entry in "${QUEUE[@]}"; do
    idx=$((idx + 1))
    IFS='|' read -r label cfg port mpath extra <<< "$entry"
    echo "[$(date)] --- [$idx/$TOTAL] $label ---" | tee -a "$QLOG"
    ntfy "A2 queue [$idx/$TOTAL]" "Starting $label — $(describe "$cfg")" "hourglass" "low"

    # shellcheck disable=SC2086  # $extra is intentionally word-split into args
    if bash run_a3_model.sh "$label" "$cfg" "$port" "$mpath" $extra >> "$QLOG" 2>&1; then
        RESULTS+=("OK   $label")
    else
        RESULTS+=("FAIL $label")
        ntfy "A2 queue: $label FAILED" "Continuing to next. Check logs/run_a2_queue.log and the per-model log." "warning" "high"
    fi
done

echo "[$(date)] === A2 queue done ===" | tee -a "$QLOG"
summary=$(printf '%s\n' "${RESULTS[@]}")
echo "$summary" | tee -a "$QLOG"
ntfy "A2 queue FINISHED" "All $TOTAL A2 runs processed:
$summary" "checkered_flag" "high"
