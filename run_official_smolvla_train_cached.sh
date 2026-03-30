#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CACHE_DIR="${HF_CACHE_DIR:-$WORKSPACE_DIR/outputs/hf_cache}"
IMAGE_NAME="${IMAGE_NAME:-lerobot-workshop:latest}"

mkdir -p "$HF_CACHE_DIR"

exec docker run --rm --gpus all \
  --shm-size=16g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_HOME=/root/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
  -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
  -v "$WORKSPACE_DIR:/app" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -w /app \
  "$IMAGE_NAME" \
  python -m lerobot.scripts.train "$@"
