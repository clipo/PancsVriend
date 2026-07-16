#!/bin/bash
# Turn experiment-log progress lines into ntfy pings.
#
#   ./watch_progress.sh /path/to/run.log &
#
# Two kinds of events:
#
# 1. STAGE STARTS — pinged immediately, so you know a scenario has begun without
#    waiting for its first run to complete:
#      "Scenario: <name>"                        (run_all_contexts, scenario loop)
#      "Running <N> simulations ..."             (llm_runner, simulation kickoff)
#      "Found incomplete/completed experiment"   (resume detection outcome)
#      "Completed workflow summary"              (all scenarios done)
#
# 2. PER-RUN PROGRESS — [run-progress] lines emitted by
#    llm_runner.format_run_progress(), one per completed run:
#      [run-progress] scenario=<s> done=<d> total=<t> elapsed_s=<e> procs=<p> \
#          [overall_done=<od> overall_total=<ot>]
#    turned into avg-runtime + ETA pings.
#
# The experiment code only LOGS; all notification transport lives here + ntfy.sh.
# Don't start this watcher (or delete it) and experiments run with zero
# notification side-effects — nothing in the runners changes.
set -u
LOG="${1:?usage: watch_progress.sh <logfile>}"
cd "$(dirname "$0")"
SCENARIO=""

tail -n 0 -F "$LOG" 2>/dev/null \
| grep -E --line-buffered '^\[run-progress\]|^Scenario: |^Running [0-9]+ simulations|^Found (incomplete|completed) experiment|^Completed workflow summary' \
| while IFS= read -r line; do
    case "$line" in

      "Scenario: "*)
        SCENARIO="${line#Scenario: }"
        continue ;;

      "Found incomplete experiment"*)
        ./ntfy.sh "${SCENARIO:-?}: resuming existing experiment" "$line" "arrows_counterclockwise" "low"
        continue ;;

      "Found completed experiment"*)
        ./ntfy.sh "${SCENARIO:-?}: already complete, reusing" "$line" "fast_forward" "low"
        continue ;;

      "Running "*" simulations"*)
        ./ntfy.sh "${SCENARIO:-?}: STARTED" "$line" "rocket" "default"
        continue ;;

      "Completed workflow summary"*)
        ./ntfy.sh "All scenarios finished" "Simulation stage done; analysis stage next." "checkered_flag" "default"
        continue ;;
    esac

    # [run-progress] k=v lines
    rest="${line#\[run-progress\] }"
    scenario=""; d=""; t=""; e=""; p="1"; od=""; ot=""
    for kv in $rest; do
      k=${kv%%=*}; v=${kv#*=}
      case "$k" in
        scenario)      scenario=$v ;;
        done)          d=$v ;;
        total)         t=$v ;;
        elapsed_s)     e=$v ;;
        procs)         p=$v ;;
        overall_done)  od=$v ;;
        overall_total) ot=$v ;;
      esac
    done
    [ -n "$d" ] && [ -n "$t" ] && [ -n "$e" ] || continue
    msg=$(awk -v e="$e" -v d="$d" -v t="$t" -v p="$p" 'BEGIN {
      avg = e / (d > 0 ? d : 1)
      eff = (p < d ? p : (d > 0 ? d : 1))
      printf "avg %.1f min/run wall (~%.1f min each at %s procs) | scenario ETA %.0f min",
             avg / 60, avg * eff / 60, p, avg * (t - d) / 60 }')
    overall=""
    [ -n "$ot" ] && overall=" | overall $od/$ot"
    ./ntfy.sh "$scenario: run $d/$t done" "$msg$overall" "hourglass_flowing_sand" "low"
  done
