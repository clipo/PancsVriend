#!/bin/bash
# Polls the gemma production log and auto-starts Mixtral when gemma completes.
set -e

cd /srv/shared/schelling/PancsVriend
source .venv/bin/activate

GEMMA_LOG="logs/run_gemma-4-31b-it-q5.log"
MIXTRAL_LOG="logs/run_mixtral8x7b.log"
TRANSITION_LOG="logs/transition_to_mixtral.log"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$TRANSITION_LOG"; }

log "Watching $GEMMA_LOG for 'Pipeline completed.'..."

while true; do
    if grep -q "Pipeline completed\." "$GEMMA_LOG" 2>/dev/null; then
        log "Pipeline completed. detected — gemma done."
        break
    fi
    log "Not done yet. Sleeping 300s..."
    sleep 300
done

# Notify gemma complete
python notify.py \
    "Schelling run complete: gemma-4-31b-it-q5" \
    "Production run finished. Results in experiments_with_llama_cpp/." || true
log "Gemma completion email sent."

# Kill gemma server
log "Killing llama_server screen..."
screen -S llama_server -X quit 2>/dev/null || true
sleep 5

# Launch Mixtral in a new screen
log "Launching Mixtral 8x7B production run..."
screen -dmS llama_run bash -c \
    "cd /srv/shared/schelling/PancsVriend && bash run_mixtral8x7b.sh 2>&1 | tee $MIXTRAL_LOG"

log "Mixtral run launched. Monitor with: tail -f $MIXTRAL_LOG"
python notify.py \
    "Schelling transition: Mixtral 8x7B starting" \
    "Gemma complete. Mixtral 8x7B Q5_K_M production run now launching. Log: $MIXTRAL_LOG" || true

log "Transition complete. Exiting."
