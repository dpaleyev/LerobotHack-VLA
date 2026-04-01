#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot v3 dataset to a lower FPS by uniform frame decimation."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Source dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Source dataset repo_id for LeRobot loading. Defaults to input folder name.",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Output dataset repo_id written into metadata. Defaults to '<repo-id>_<target-fps>fps'.",
    )
    parser.add_argument("--target-fps", type=int, default=10, help="Target FPS.")
    parser.add_argument(
        "--episode-start",
        type=int,
        default=0,
        help="First episode index to convert (inclusive).",
    )
    parser.add_argument(
        "--episode-stop",
        type=int,
        default=None,
        help="Last episode index to convert (exclusive). Defaults to all remaining episodes.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs.",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        help="LeRobot video backend for reading source video datasets.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="libsvtav1",
        help="Video codec for writing output video datasets.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Passed through to LeRobotDataset.create().",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Passed through to LeRobotDataset.create().",
    )
    parser.add_argument(
        "--streaming-encoding",
        action="store_true",
        help="Use LeRobot streaming video encoder for output video datasets.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Threads per output encoder instance.",
    )
    parser.add_argument(
        "--metadata-buffer-size",
        type=int,
        default=10,
        help="Episode metadata buffer size for the output dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be converted without writing files.",
    )
    return parser.parse_args()


def strip_auto_features(features: dict[str, dict]) -> dict[str, dict]:
    return {key: copy.deepcopy(value) for key, value in features.items() if key not in DEFAULT_FEATURES}


def prepare_output_features(features: dict[str, dict]) -> dict[str, dict]:
    output_features = strip_auto_features(features)
    for key, spec in output_features.items():
        if spec["dtype"] == "video":
            spec.pop("info", None)
    return output_features


def select_episodes(total_episodes: int, start: int, stop: int | None, max_episodes: int | None) -> list[int]:
    if start < 0:
        raise SystemExit("--episode-start must be non-negative.")
    stop = total_episodes if stop is None else min(stop, total_episodes)
    if stop < start:
        raise SystemExit("--episode-stop must be >= --episode-start.")
    episode_indices = list(range(start, stop))
    if max_episodes is not None:
        if max_episodes <= 0:
            raise SystemExit("--max-episodes must be positive.")
        episode_indices = episode_indices[:max_episodes]
    if not episode_indices:
        raise SystemExit("No episodes selected for conversion.")
    return episode_indices


def adapt_numeric_value(value: object, dtype: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value, dtype=np.dtype(dtype))

    if not isinstance(value, np.ndarray):
        value = np.asarray(value, dtype=np.dtype(dtype))

    if value.dtype != np.dtype(dtype):
        value = value.astype(dtype, copy=False)

    if value.shape == () and expected_shape == (1,):
        value = value.reshape(1)

    if value.shape != expected_shape:
        value = np.asarray(value, dtype=np.dtype(dtype)).reshape(expected_shape)

    return value


def adapt_visual_value(value: object, expected_shape: tuple[int, int, int]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value)

    if not isinstance(value, np.ndarray):
        value = np.asarray(value)

    if value.ndim != 3:
        raise ValueError(f"Visual feature must be 3D, got shape {value.shape}.")

    height, width, channels = expected_shape
    if value.shape == (channels, height, width):
        value = np.transpose(value, (1, 2, 0))
    elif value.shape != expected_shape:
        raise ValueError(
            f"Unexpected visual feature shape {value.shape}; expected {expected_shape} or {(channels, height, width)}."
        )

    return value


def build_frame(item: dict, features: dict[str, dict]) -> dict:
    frame = {"task": item["task"]}
    for key, spec in features.items():
        if key in DEFAULT_FEATURES:
            continue

        value = item[key]
        if spec["dtype"] in {"image", "video"}:
            frame[key] = adapt_visual_value(value, tuple(spec["shape"]))
            continue

        if spec["dtype"] == "string":
            frame[key] = str(value)
            continue

        frame[key] = adapt_numeric_value(value, spec["dtype"], tuple(spec["shape"]))

    return frame


def copy_optional_subtasks(source_root: Path, target_root: Path) -> None:
    source = source_root / "meta" / "subtasks.parquet"
    target = target_root / "meta" / "subtasks.parquet"
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args()

    repo_id = args.repo_id or args.input_root.name
    source = LeRobotDataset(
        repo_id=repo_id,
        root=args.input_root,
        video_backend=args.video_backend,
    )

    source_fps = int(source.fps)
    target_fps = int(args.target_fps)
    if target_fps <= 0:
        raise SystemExit("--target-fps must be positive.")
    if target_fps > source_fps:
        raise SystemExit(f"Target FPS ({target_fps}) cannot exceed source FPS ({source_fps}).")
    if source_fps % target_fps != 0:
        raise SystemExit(
            f"Only integer decimation is supported. Source FPS {source_fps} is not divisible by target FPS {target_fps}."
        )

    step = source_fps // target_fps
    episode_indices = select_episodes(
        total_episodes=source.meta.total_episodes,
        start=args.episode_start,
        stop=args.episode_stop,
        max_episodes=args.max_episodes,
    )

    sampled_lengths: list[int] = []
    for ep_idx in episode_indices:
        ep = source.meta.episodes[ep_idx]
        ep_length = int(ep["length"])
        sampled_lengths.append(len(range(0, ep_length, step)))

    print(f"Source dataset: {args.input_root}")
    print(f"Source FPS: {source_fps}")
    print(f"Target FPS: {target_fps}")
    print(f"Frame step: {step}")
    print(f"Selected episodes: {len(episode_indices)}")
    print(f"Expected output frames: {sum(sampled_lengths)}")
    print(f"Visual storage: {'video' if source.meta.video_keys else 'image'}")

    if args.dry_run:
        return

    if args.output_root.exists():
        raise SystemExit(f"Output already exists: {args.output_root}")

    output_repo_id = args.output_repo_id or f"{repo_id}_{target_fps}fps"
    output_features = prepare_output_features(source.meta.features)
    output = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=target_fps,
        features=output_features,
        root=args.output_root,
        robot_type=source.meta.robot_type,
        use_videos=bool(source.meta.video_keys),
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=1,
        vcodec=args.vcodec,
        metadata_buffer_size=args.metadata_buffer_size,
        streaming_encoding=args.streaming_encoding,
        encoder_threads=args.encoder_threads,
    )
    output.meta.update_chunk_settings(
        chunks_size=source.meta.chunks_size,
        data_files_size_in_mb=source.meta.data_files_size_in_mb,
        video_files_size_in_mb=source.meta.video_files_size_in_mb,
    )

    try:
        progress = tqdm(
            episode_indices,
            desc=f"Converting {source_fps}fps -> {target_fps}fps",
            unit="episode",
        )
        for ep_idx in progress:
            episode = source.meta.episodes[ep_idx]
            start = int(episode["dataset_from_index"])
            stop = int(episode["dataset_to_index"])
            sampled_indices = range(start, stop, step)

            kept_frames = 0
            for frame_idx in sampled_indices:
                item = source[frame_idx]
                output.add_frame(build_frame(item, source.meta.features))
                kept_frames += 1

            if kept_frames == 0:
                raise RuntimeError(f"Episode {ep_idx} produced no frames after decimation.")

            output.save_episode(parallel_encoding=True)
            progress.set_postfix(ep=ep_idx, kept_frames=kept_frames)
    finally:
        output.finalize()

    copy_optional_subtasks(source.root, output.root)

    print(f"Converted dataset saved to: {output.root}")
    print(f"Output repo_id: {output.repo_id}")
    print(f"Output episodes: {output.meta.total_episodes}")
    print(f"Output frames: {output.meta.total_frames}")


if __name__ == "__main__":
    main()
