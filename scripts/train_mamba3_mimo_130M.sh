#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=mamba3_mimo SCALE=130M exec "${SCRIPT_DIR}/train.sh" "$@"
