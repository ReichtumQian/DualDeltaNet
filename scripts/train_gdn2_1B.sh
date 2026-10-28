#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=gdn2 SCALE=1B exec "${SCRIPT_DIR}/train.sh" "$@"
