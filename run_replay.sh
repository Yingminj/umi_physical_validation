#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 && -z "${SCORE_FOLDER_PATH:-}" ]]; then
  echo "Usage: $0 <episode_dir> [trajectory_index] [replay options]" >&2
  echo "Or set SCORE_FOLDER_PATH before running." >&2
  exit 2
fi

if [[ $# -ge 1 ]]; then
  export SCORE_FOLDER_PATH="$1"
fi

trajectory_index="${2:-0}"
if [[ $# -ge 2 ]]; then
  shift 2
elif [[ $# -eq 1 ]]; then
  shift 1
fi
export ROBOT_NAME="${ROBOT_NAME:-robot_a}"
export PYTHONUNBUFFERED=1

cd "${SCRIPT_DIR}"
exec python3 replay.py "${trajectory_index}" "$@"
