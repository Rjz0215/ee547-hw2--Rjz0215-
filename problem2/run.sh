#!/bin/bash
set -euo pipefail

# Accept input file and output directory
INPUT_PATH=${1:-../problem1/sample_data/papers.json}
OUTPUT_DIR=${2:-output}
EPOCHS="${3:-50}"
BATCH_SIZE="${4:-32}"

# Create output directory on host
mkdir -p "$OUTPUT_DIR"

# Resolve absolute paths for mounting
# On Linux/macOS use realpath; on Windows Git-Bash use pwd -W
if [[ "$(uname -s)" =~ MINGW|MSYS|CYGWIN ]]; then
  INPUT_REAL="$(cd "$(dirname "$INPUT_PATH")" && pwd -W)/$(basename "$INPUT_PATH")"
  OUTPUT_REAL="$(cd "$OUTPUT_DIR" && pwd -W)"
else
  INPUT_REAL="$(realpath "$INPUT_PATH")"
  OUTPUT_REAL="$(realpath "$OUTPUT_DIR")"
fi

echo "Training embeddings with the following settings:"
echo "  Input: $INPUT_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Run training container
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" docker run --rm \
  --name arxiv-embeddings \
  -v "$INPUT_REAL":/app/data/input/papers.json:ro \
  -v "$OUTPUT_REAL":/app/output \
  arxiv-embeddings:latest \
  /app/data/input/papers.json /app/output \
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"

echo ""
echo "Training complete. Output files:"
ls -la "$OUTPUT_DIR"
