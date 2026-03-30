#!/usr/bin/env python
"""
Fine-tune SmolVLA on collected SO-101 MuJoCo demonstrations.

Usage:
  python train_smolvla.py
  python train_smolvla.py --pretrained lerobot/smolvla_base
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from smolvla_compat import load_smolvla_config
from smolvla_defaults import (
    DEFAULT_DATASET_REPO_ID,
    DEFAULT_PRETRAINED_POLICY,
    default_dataset_root,
    default_output_dir,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", type=Path, default=default_dataset_root())
    p.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED_POLICY)
    p.add_argument("--output-dir", type=Path, default=default_output_dir())
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--save-every", type=int, default=2_000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--n-action-steps", type=int, default=None)
    p.add_argument("--empty-cameras", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--amp", choices=["auto", "off", "bf16"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def save_checkpoint(policy: SmolVLAPolicy, output_dir: Path, step: int):
    ckpt_dir = output_dir / f"step_{step}" / "pretrained_model"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(ckpt_dir))
    last_dir = output_dir / "checkpoints" / "last" / "pretrained_model"
    if last_dir.exists():
        shutil.rmtree(last_dir)
    last_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(last_dir))
    log.info("Saved checkpoint at step %s -> %s", step, ckpt_dir)


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def get_amp_dtype(device: torch.device, amp_mode: str) -> torch.dtype | None:
    if device.type != "cuda" or amp_mode == "off":
        return None
    if amp_mode == "bf16":
        return torch.bfloat16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def build_policy_config(
    pretrained: str,
    input_features: dict,
    output_features: dict,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    chunk_size: int | None,
    n_action_steps: int | None,
    empty_cameras: int | None,
    lr: float | None,
    warmup_steps: int | None,
    grad_clip: float | None,
) -> SmolVLAConfig:
    cfg = load_smolvla_config(pretrained, device=device)
    cfg.input_features = input_features
    cfg.output_features = output_features
    cfg.use_amp = amp_dtype is not None

    if chunk_size is not None:
        cfg.chunk_size = chunk_size
    if n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    if empty_cameras is not None:
        cfg.empty_cameras = empty_cameras
    if lr is not None:
        cfg.optimizer_lr = lr
    if warmup_steps is not None:
        cfg.scheduler_warmup_steps = warmup_steps
    if grad_clip is not None:
        cfg.optimizer_grad_clip_norm = grad_clip

    cfg.__post_init__()
    cfg.validate_features()
    return cfg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    log.info("Loading dataset from %s", args.dataset_path)
    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=str(args.dataset_path))
    features = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    amp_dtype = get_amp_dtype(device, args.amp)
    cfg = build_policy_config(
        pretrained=args.pretrained,
        input_features=input_features,
        output_features=output_features,
        device=device,
        amp_dtype=amp_dtype,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        empty_cameras=args.empty_cameras,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
    )

    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=str(args.dataset_path),
        delta_timestamps=delta_timestamps,
    )
    log.info("Dataset: %s frames, %s episodes", dataset.num_frames, dataset.num_episodes)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        drop_last=True,
    )

    log.info("Loading SmolVLA from %s", args.pretrained)
    policy = SmolVLAPolicy.from_pretrained(
        args.pretrained,
        config=cfg,
        dataset_stats=meta.stats,
    )
    if device.type != "cuda":
        policy.float()
    policy.to(device)
    policy.train()

    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    log.info("Parameters: %s trainable / %s total", f"{n_trainable:,}", f"{n_total:,}")

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer_cfg = cfg.get_optimizer_preset()
    optimizer = optimizer_cfg.build(trainable_params)
    scheduler_cfg = cfg.get_scheduler_preset()
    scheduler = scheduler_cfg.build(optimizer, args.steps) if scheduler_cfg is not None else None

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dl_iter = cycle(dataloader)
    running_loss = 0.0
    t0 = time.time()

    log.info(
        "Training for %s steps, batch_size=%s, lr=%s, empty_cameras=%s, num_workers=%s, amp=%s",
        args.steps,
        args.batch_size,
        optimizer.param_groups[0]["lr"],
        cfg.empty_cameras,
        args.num_workers,
        amp_dtype if amp_dtype is not None else "off",
    )
    progress = tqdm(total=args.steps, desc="train", dynamic_ncols=True)
    try:
        for step in range(1, args.steps + 1):
            batch = next(dl_iter)
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, _ = policy.forward(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, optimizer_cfg.grad_clip_norm, error_if_nonfinite=False
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            progress.update(1)

            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                elapsed = time.time() - t0
                step_per_s = step / elapsed
                progress.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    grad=f"{grad_norm.item():.2f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    steps=f"{step_per_s:.2f}/s",
                )
                running_loss = 0.0

            if step % args.save_every == 0 or step == args.steps:
                policy.eval()
                save_checkpoint(policy, output_dir, step)
                policy.train()
                progress.write(f"Saved checkpoint at step {step}")
    finally:
        progress.close()

    log.info("Training complete!")
    log.info("Final checkpoint: %s", output_dir / "checkpoints" / "last" / "pretrained_model")


if __name__ == "__main__":
    main()
