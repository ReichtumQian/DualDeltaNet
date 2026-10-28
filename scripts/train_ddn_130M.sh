#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=ddn SCALE=130M exec "${SCRIPT_DIR}/train.sh" "$@"
