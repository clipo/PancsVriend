#!/bin/bash
# Resolve a GGUF filename to its absolute path.
# Search order:
#   1. /srv/shared/schelling/PancsVriend/llms/  (project copy)
#   2. ~/llms/                                   (user home copy)
#   3. the argument itself if it is already an absolute path

FILENAME="$1"

if [ -z "$FILENAME" ]; then
    echo "Usage: find_gguf.sh <filename-or-path>" >&2
    exit 1
fi

PROJECT_LLMS="/srv/shared/schelling/PancsVriend/llms"
HOME_LLMS="$HOME/llms"

BASENAME="$(basename "$FILENAME")"

if [ -f "$PROJECT_LLMS/$BASENAME" ]; then
    echo "$PROJECT_LLMS/$BASENAME"
elif [ -f "$HOME_LLMS/$BASENAME" ]; then
    echo "$HOME_LLMS/$BASENAME"
elif [ -f "$FILENAME" ]; then
    echo "$FILENAME"
else
    echo "ERROR: '$BASENAME' not found in $PROJECT_LLMS or $HOME_LLMS" >&2
    exit 1
fi
