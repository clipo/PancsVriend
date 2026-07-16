#!/bin/bash
# Push experiment status to the ntfy app (phone). Independent of notify.py (email).
#
#   ./ntfy.sh "Title" "Message body" [tags] [priority]
#
# Topic override:  export NTFY_TOPIC=your-topic
# Subscribe to the topic in the ntfy app to receive these.
# NOTE: the topic name is the ONLY access control — anyone who knows it can read.

NTFY_TOPIC="${NTFY_TOPIC:-schelling-runs-7f3a9c}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"

title="${1:-PancsVriend}"
body="${2:-}"
tags="${3:-test_tube}"
prio="${4:-default}"

curl -fsS \
  -H "Title: ${title}" \
  -H "Tags: ${tags}" \
  -H "Priority: ${prio}" \
  -d "${body}" \
  "${NTFY_SERVER}/${NTFY_TOPIC}" > /dev/null \
  && echo "[ntfy] sent: ${title}" \
  || echo "[ntfy] FAILED to send: ${title}" >&2
