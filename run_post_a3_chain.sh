#!/bin/bash
# Post-A3 chain (supersedes run_a2_after_a3.sh):
#   1. wait for the A3 queue (run_a3_queue.sh) to finish
#   2. run the ADDED Gemma A3 chat+grammar run (so Gemma has both endpoints
#      under A3, alongside its completions+grammar run in the A3 queue)
#   3. run the A2 queue (run_a2_queue.sh)
#
# Launch detached:
#     nohup bash run_post_a3_chain.sh >> logs/run_post_a3_chain.log 2>&1 &
#
# pgrep uses a char-class first letter so it never matches THIS script's own
# command line (which contains run_post_a3_chain.sh, not the A3 queue name).
set -u
cd /srv/shared/schelling/PancsVriend
LOG="logs/run_post_a3_chain.log"
mkdir -p logs

echo "[$(date)] waiting for A3 queue (run_a3_queue.sh) to finish..." | tee -a "$LOG"
for _ in $(seq 1 10); do
    pgrep -f "[r]un_a3_queue.sh" >/dev/null 2>&1 && break
    sleep 6
done
while pgrep -f "[r]un_a3_queue.sh" >/dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] A3 queue done → added Gemma A3 chat+grammar run" | tee -a "$LOG"
./ntfy.sh "A3 done → Gemma a3-chat" "Running the added Gemma A3 chat+grammar run (10 runs), then the A2 batch." "arrow_forward" "default"
if bash run_a3_model.sh gemma-4-31b-a3-chat \
        configs/llama_cpp_run_gemma31b_a3_chat.yaml \
        8081 llms/gemma-4-31B-it-Q5_K_M.gguf \
        --jinja --reasoning off >> "$LOG" 2>&1; then
    echo "[$(date)] Gemma a3-chat OK" | tee -a "$LOG"
else
    echo "[$(date)] Gemma a3-chat FAILED (continuing to A2)" | tee -a "$LOG"
    ./ntfy.sh "Gemma a3-chat FAILED" "Continuing to the A2 batch. Check $LOG." "warning" "high"
fi

echo "[$(date)] → A2 queue" | tee -a "$LOG"
exec bash run_a2_queue.sh
