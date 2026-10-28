#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=gdn2 SCALE=340M exec "${SCRIPT_DIR}/train.sh" "$@"
