#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CACHE_DIR="${HF_CACHE_DIR:-$WORKSPACE_DIR/outputs/hf_cache}"
IMAGE_NAME="${IMAGE_NAME:-lerobot-workshop:latest}"
DEFAULT_PUSH_TO_HUB="${DEFAULT_PUSH_TO_HUB:-false}"
DEFAULT_VIDEO_BACKEND="${DEFAULT_VIDEO_BACKEND:-pyav}"
DEFAULT_POLICY_EMPTY_CAMERAS="${DEFAULT_POLICY_EMPTY_CAMERAS:-1}"
DEFAULT_RENAME_MAP="${DEFAULT_RENAME_MAP:-}"

if [[ -z "$DEFAULT_RENAME_MAP" ]]; then
  DEFAULT_RENAME_MAP='{"observation.images.front":"observation.images.camera1","observation.images.side":"observation.images.camera2"}'
fi

mkdir -p "$HF_CACHE_DIR"

default_args=(
  --policy.push_to_hub="$DEFAULT_PUSH_TO_HUB"
  --dataset.video_backend="$DEFAULT_VIDEO_BACKEND"
)

if [[ -n "$DEFAULT_POLICY_EMPTY_CAMERAS" ]]; then
  default_args+=(--policy.empty_cameras="$DEFAULT_POLICY_EMPTY_CAMERAS")
fi

if [[ -n "$DEFAULT_RENAME_MAP" ]]; then
  default_args+=(--rename_map="$DEFAULT_RENAME_MAP")
fi

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
  lerobot-train "${default_args[@]}" "$@"
