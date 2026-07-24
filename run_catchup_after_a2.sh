#!/bin/bash
# Waits for the in-flight A2 queue to finish, then runs the chat catch-up queue.
# (run_a2_queue.sh is already executing a fixed list and cannot be edited
# mid-loop, so the catch-up is chained behind it rather than inserted.)
#
# Launch detached:
#     nohup bash run_catchup_after_a2.sh >> logs/run_catchup_after_a2.log 2>&1 &
#
# The pgrep pattern uses a char class on the first letter so it can never match
# this script's own command line.
set -u
cd /srv/shared/schelling/PancsVriend
LOG="logs/run_catchup_after_a2.log"
mkdir -p logs

echo "[$(date)] waiting for run_a2_queue.sh to finish..." | tee -a "$LOG"
while pgrep -f "[r]un_a2_queue.sh" >/dev/null 2>&1; do
    sleep 60
done

echo "[$(date)] A2 queue done -> chat catch-up queue" | tee -a "$LOG"
exec bash run_chat_catchup_queue.sh
