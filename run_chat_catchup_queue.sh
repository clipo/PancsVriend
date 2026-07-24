#!/bin/bash
# Catch-up queue: re-runs every chat+grammar run that died on the pre-fix
# connection probe.
#
# All four failed identically and within ~1s of the server coming up:
# check_llm_connection() hardcoded a raw-completions "prompt" payload, so the
# /v1/chat/completions endpoint answered HTTP 400 ('messages' is required) and
# the run aborted before a single simulation step. Fixed in llm_runner.py by
# building the probe with build_llm_request(llm_style=...), i.e. the same
# builder the simulation itself uses.
#
# Order restores the original campaign intent (A3 first, then A2):
#   1. llama-3.3-70b  A3 chat+grammar  (8082, 10 runs)
#   2. qwen3.6-27b    A3 chat+grammar  (8084, 10 runs)
#   3. gemma-4-31b    A3 chat+grammar  (8081, 10 runs)
#   4. gemma-4-31b    A2 chat+grammar  (8081,  5 runs)
#
# The other two chat runs still queued in run_a2_queue.sh (llama-a2-chat,
# qwen-a2-chat) are NOT repeated here: they have not started yet and each
# run_a3_model.sh invocation launches a fresh interpreter, so they pick the
# fix up on their own.
#
#   ./run_chat_catchup_queue.sh    (long-running; launch in the background)
set -u

cd /srv/shared/schelling/PancsVriend
QLOG="logs/run_chat_catchup_queue.log"
mkdir -p logs
PYBIN=".venv/bin/python"

ntfy() { ./ntfy.sh "$1" "$2" "${3:-test_tube}" "${4:-default}"; }
# One-line experiment summary derived from the run YAML (never hardcode counts).
describe() { "$PYBIN" notifications.py describe --config "$1" 2>/dev/null || echo "?"; }

# label | run_cfg | port | model_path | extra llama-server args
QUEUE=(
  "llama-3.3-70b-a3-chat|configs/llama_cpp_run_llama70b_a3_chat.yaml|8082|llms/Llama-3.3-70B-Instruct-Q4_K_M.gguf|--jinja --reasoning off"
  "qwen3.6-27b-a3-chat|configs/llama_cpp_run_qwen36_27b_a3_chat.yaml|8084|llms/Qwen3.6-27B-Q5_K_M.gguf|--jinja --reasoning off"
  "gemma-4-31b-a3-chat|configs/llama_cpp_run_gemma31b_a3_chat.yaml|8081|llms/gemma-4-31B-it-Q5_K_M.gguf|--jinja --reasoning off"
  "gemma-4-31b-a2-chat|configs/llama_cpp_run_gemma31b_a2_chat.yaml|8081|llms/gemma-4-31B-it-Q5_K_M.gguf|--jinja --reasoning off"
)

TOTAL=${#QUEUE[@]}
LINEUP=""
for entry in "${QUEUE[@]}"; do
    IFS='|' read -r label cfg _ _ _ <<< "$entry"
    LINEUP+="$label: $(describe "$cfg")"$'\n'
done
echo "[$(date)] === chat catch-up queue starting: $TOTAL runs ===" | tee -a "$QLOG"
printf '%s' "$LINEUP" | tee -a "$QLOG"
ntfy "Chat catch-up STARTED" "$TOTAL re-runs of the chat+grammar runs that died on the old HTTP 400 probe:
$LINEUP" "arrow_forward" "default"

declare -a RESULTS
idx=0
for entry in "${QUEUE[@]}"; do
    idx=$((idx + 1))
    IFS='|' read -r label cfg port mpath extra <<< "$entry"
    echo "[$(date)] --- [$idx/$TOTAL] $label ---" | tee -a "$QLOG"
    ntfy "Catch-up [$idx/$TOTAL]" "Starting $label — $(describe "$cfg")" "hourglass" "low"

    # shellcheck disable=SC2086  # $extra is intentionally word-split into args
    if bash run_a3_model.sh "$label" "$cfg" "$port" "$mpath" $extra >> "$QLOG" 2>&1; then
        RESULTS+=("OK   $label")
    else
        RESULTS+=("FAIL $label")
        ntfy "Catch-up: $label FAILED" "Continuing to next. Check logs/run_chat_catchup_queue.log and the per-model log." "warning" "high"
    fi
done

echo "[$(date)] === chat catch-up queue done ===" | tee -a "$QLOG"
summary=$(printf '%s\n' "${RESULTS[@]}")
echo "$summary" | tee -a "$QLOG"
ntfy "Chat catch-up FINISHED" "All $TOTAL catch-up runs processed:
$summary" "checkered_flag" "high"
