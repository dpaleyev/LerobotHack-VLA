#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CACHE_DIR="${HF_CACHE_DIR:-$WORKSPACE_DIR/outputs/hf_cache}"
# Must match where checkpoints live; run_smolvla_inference.py defaults to ...noamp (no _2).
TRAIN_RUN_REL="${TRAIN_RUN_REL:-outputs/train/so101_smolvla_official_main_bs32_lr1e4_noamp_2}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-$WORKSPACE_DIR/$TRAIN_RUN_REL}"
# Inside Docker (-w /app) we must pass a path under the mount, not the host absolute path.
TRAIN_RUN_DOCKER="$(realpath --relative-to="$WORKSPACE_DIR" "$TRAIN_RUN_DIR")"
IMAGE_NAME="${IMAGE_NAME:-lerobot-workshop:latest}"
CHECKPOINTS="${CHECKPOINTS:-1000 2000 3000 4000 5000 6000 7000}"
EPISODES="${EPISODES:-20}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEED="${SEED:-42}"
OUT_DIR="$WORKSPACE_DIR/outputs/eval/benchmark_${EPISODES}scenes_steps${MAX_STEPS}"

mkdir -p "$OUT_DIR"
mkdir -p "$HF_CACHE_DIR"

echo "============================================================"
echo "Checkpoint benchmark"
echo "  workspace:   $WORKSPACE_DIR"
echo "  image:       $IMAGE_NAME"
echo "  episodes:    $EPISODES"
echo "  max_steps:   $MAX_STEPS"
echo "  seed:        $SEED"
echo "  DISPLAY:     ${DISPLAY:-<not set>}"
echo "  out_dir:     $OUT_DIR"
echo "  train_run:   $TRAIN_RUN_DIR"
echo "  checkpoints: $CHECKPOINTS"
echo "============================================================"

for STEP in $CHECKPOINTS; do
    STEP_PAD=$(printf "%06d" "$STEP")
    CKPT_PATH="$TRAIN_RUN_DIR/checkpoints/$STEP_PAD/pretrained_model"
    SUMMARY_PATH="/app/outputs/eval/benchmark_${EPISODES}scenes_steps${MAX_STEPS}/ckpt_${STEP_PAD}.json"
    LOCAL_SUMMARY="$OUT_DIR/ckpt_${STEP_PAD}.json"

    echo ""
    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] === Checkpoint $STEP ==="
    echo "  ckpt_path: $CKPT_PATH"

    if [ ! -d "$CKPT_PATH" ]; then
        echo "  SKIP: directory not found"
        continue
    fi

    DISPLAY_ARGS=()
    if [ -n "${DISPLAY:-}" ]; then
        DISPLAY_ARGS=(
            -e "DISPLAY=$DISPLAY"
            -v "/tmp/.X11-unix:/tmp/.X11-unix"
        )
        echo "  display:   $DISPLAY (X11 forwarded)"
    else
        echo "  display:   headless (no DISPLAY set)"
        DISPLAY_ARGS=(-e "MUJOCO_GLFW_VISIBLE=0")
    fi

    docker run --rm --gpus all \
        --shm-size=16g \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e HF_HOME=/root/.cache/huggingface \
        -e HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub \
        -e HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
        -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
        "${DISPLAY_ARGS[@]}" \
        -v "$WORKSPACE_DIR:/app" \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
        -w /app \
        "$IMAGE_NAME" \
        python run_smolvla_inference.py \
            --train-run-dir "$TRAIN_RUN_DOCKER" \
            --checkpoint-step "$STEP" \
            --episodes "$EPISODES" \
            --max-steps "$MAX_STEPS" \
            --seed "$SEED" \
            --device cuda \
            --summary-path "$SUMMARY_PATH"

    RC=$?
    if [ $RC -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] ERROR: checkpoint $STEP failed with exit code $RC"
        exit $RC
    fi

    if [ -f "$LOCAL_SUMMARY" ]; then
        SR=$(python3 -c "
import json, sys
d = json.load(open('$LOCAL_SUMMARY'))
eps = [(r['success'], r['steps']) for r in d.get('results', [])]
for i, (ok, s) in enumerate(eps):
    flag = 'OK ' if ok else 'TO '
    print(f'  ep {i:02d}: {flag}  steps={s}')
print(f\"  TOTAL: {d['successes']}/{d['episodes']}  sr={d['success_rate']:.3f}  avg_steps={d['avg_steps']:.1f}\")
")
        echo "[$(date '+%H:%M:%S')] checkpoint $STEP done:"
        echo "$SR"
    else
        echo "[$(date '+%H:%M:%S')] WARNING: summary file not found after run"
    fi
done

echo ""
echo "============================================================"
echo "All checkpoints done. Building summary table..."
echo "============================================================"

python3 - <<PY
import json
from pathlib import Path

episodes = int("$EPISODES")
max_steps = int("$MAX_STEPS")
out_dir = Path("$OUT_DIR")
checkpoints = [int(s) for s in "$CHECKPOINTS".split()]
rows = []

for step in checkpoints:
    p = out_dir / f"ckpt_{step:06d}.json"
    if not p.exists():
        print(f"  MISSING: {p}")
        continue
    d = json.loads(p.read_text())
    rows.append({
        "checkpoint": step,
        "successes": d["successes"],
        "episodes": d["episodes"],
        "success_rate": d["success_rate"],
        "avg_steps": d["avg_steps"],
    })

summary = {"episodes": episodes, "max_steps": max_steps, "results": rows}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "| checkpoint | successes | success_rate | avg_steps |",
    "|---:|:---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['checkpoint']} | {r['successes']}/{r['episodes']} | {r['success_rate']:.3f} | {r['avg_steps']:.1f} |"
    )
md = "\n".join(lines) + "\n"
(out_dir / "summary.md").write_text(md)
print(md)
PY
