#!/usr/bin/env bash
set -e

INPUT_DIR=${1:-data/input}
OUTPUT_ZIP=${2:-output.zip}
CONFIG=${3:-configs/llm_pseudo_hybrid.yaml}

python -m src.main \
  --input_dir "$INPUT_DIR" \
  --output_zip "$OUTPUT_ZIP" \
  --config "$CONFIG"
