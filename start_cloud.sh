#!/usr/bin/env bash
set -euo pipefail

export SUBTITLE_STUDIO_MODE="${SUBTITLE_STUDIO_MODE:-cloud}"
export SUBTITLE_STUDIO_DATA_DIR="${SUBTITLE_STUDIO_DATA_DIR:-/root/autodl-tmp/subtitle-studio-data}"

STUDIO_HOST="${SUBTITLE_STUDIO_HOST:-127.0.0.1}"
STUDIO_PORT="${SUBTITLE_STUDIO_PORT:-6006}"
mkdir -p "$SUBTITLE_STUDIO_DATA_DIR"

python -m uvicorn studio.main:app --host "$STUDIO_HOST" --port "$STUDIO_PORT"
