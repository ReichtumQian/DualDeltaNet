#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=mamba3_siso SCALE=1B exec "${SCRIPT_DIR}/train.sh" "$@"
