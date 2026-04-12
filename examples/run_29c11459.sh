#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Run the interpreter-executor solver on ARC task 2204b7a8
#
# Prerequisites:
#   export OPENROUTER_API_KEY="sk-..."
#   pip install -r requirements.txt
#   Add model config to arc/llm/config.py
# ──────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TASK_ID="29c11459"
DATA_FOLDER="data/arc-prize-2024"
INSTRUCTION="tasks/${TASK_ID}/instruction.txt"
LOG_DIR="results/${TASK_ID}"
REPORT="${LOG_DIR}/report.pdf"
OUTPUT="${LOG_DIR}/output.json"
MODEL_KEY="gemma-3-12b-backup"
LOG_JSON="${LOG_DIR}/${TASK_ID}_log.json"

mkdir -p "$LOG_DIR"

python -m arc \
    --task_id "$TASK_ID" \
    --data_folder "$DATA_FOLDER" \
    --instruction "$INSTRUCTION" \
    --model_key "$MODEL_KEY" \
    --log_dir "$LOG_DIR" \
    --report "$REPORT" \
    --output "$OUTPUT" \
    --max_steps 500 \
    --time_sleep 10 \
    --log_every 5 \
    --resume "$LOG_JSON" \
    "$@"

echo ""
echo "Done. Results in ${LOG_DIR}/"
