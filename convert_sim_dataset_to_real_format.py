#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a simulation LeRobot dataset to the same schema as real SO-101 datasets."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Source simulation dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output compatible dataset root.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Source dataset repo_id. Defaults to the input folder name.",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Output repo_id written into metadata. Defaults to '<repo-id>_realfmt'.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="so_follower",
        help="Target robot_type for the converted dataset.",
    )
    parser.add_argument(
        "--front-source-key",
        type=str,
        default="observation.image",
        help="Source key to map into observation.images.front.",
    )
    parser.add_argument(
        "--side-source-key",
        type=str,
        default="observation.wrist_image",
        help="Source key to map into observation.images.side.",
    )
    parser.add_argument(
        "--state-source-key",
        type=str,
        default="observation.state",
        help="Source key for robot state.",
    )
    parser.add_argument(
        "--action-source-key",
        type=str,
        default="action",
        help="Source key for action.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        help="Video codec for the output dataset.",
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
        help="Use the LeRobot streaming video encoder for output videos.",
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
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be converted without writing files.",
    )
    return parser.parse_args()


def adapt_numeric_value(value: object, dtype: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
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


def build_output_features(source_features: dict[str, dict], front_source_key: str, side_source_key: str) -> dict[str, dict]:
    front_spec = source_features[front_source_key]
    side_spec = source_features[side_source_key]
    state_spec = source_features["observation.state"]
    action_spec = source_features["action"]

    return {
        "action": {
            "dtype": action_spec["dtype"],
            "shape": action_spec["shape"],
            "names": JOINT_NAMES,
        },
        "observation.state": {
            "dtype": state_spec["dtype"],
            "shape": state_spec["shape"],
            "names": JOINT_NAMES,
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": front_spec["shape"],
            "names": front_spec["names"],
        },
        "observation.images.side": {
            "dtype": "video",
            "shape": side_spec["shape"],
            "names": side_spec["names"],
        },
    }


def build_frame(item: dict, output_features: dict[str, dict], args: argparse.Namespace) -> dict:
    frame = {"task": item["task"]}
    frame["action"] = adapt_numeric_value(
        item[args.action_source_key],
        output_features["action"]["dtype"],
        tuple(output_features["action"]["shape"]),
    )
    frame["observation.state"] = adapt_numeric_value(
        item[args.state_source_key],
        output_features["observation.state"]["dtype"],
        tuple(output_features["observation.state"]["shape"]),
    )
    frame["observation.images.front"] = adapt_visual_value(
        item[args.front_source_key],
        tuple(output_features["observation.images.front"]["shape"]),
    )
    frame["observation.images.side"] = adapt_visual_value(
        item[args.side_source_key],
        tuple(output_features["observation.images.side"]["shape"]),
    )
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
    output_repo_id = args.output_repo_id or f"{repo_id}_realfmt"

    source = LeRobotDataset(repo_id=repo_id, root=args.input_root)

    required_keys = {
        args.front_source_key,
        args.side_source_key,
        args.state_source_key,
        args.action_source_key,
    }
    missing = sorted(key for key in required_keys if key not in source.meta.features)
    if missing:
        raise SystemExit(f"Source dataset is missing required features: {missing}")

    if args.output_root.exists():
        raise SystemExit(f"Output already exists: {args.output_root}")

    total_episodes = source.meta.total_episodes
    episode_indices = list(range(total_episodes))
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise SystemExit("--max-episodes must be positive.")
        episode_indices = episode_indices[: args.max_episodes]

    output_features = build_output_features(
        source.meta.features,
        front_source_key=args.front_source_key,
        side_source_key=args.side_source_key,
    )
    if args.dry_run:
        print(f"Source dataset: {args.input_root}")
        print(f"Output dataset: {args.output_root}")
        print(f"Source repo_id: {repo_id}")
        print(f"Output repo_id: {output_repo_id}")
        print(f"Source robot_type: {source.meta.robot_type}")
        print(f"Target robot_type: {args.robot_type}")
        print(f"Selected episodes: {len(episode_indices)} / {total_episodes}")
        print(f"Output features: {list(output_features)}")
        return

    output = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=source.meta.fps,
        features={k: v for k, v in output_features.items() if k not in DEFAULT_FEATURES},
        root=args.output_root,
        robot_type=args.robot_type,
        use_videos=True,
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
        progress = tqdm(episode_indices, desc="Converting sim -> real format", unit="episode")
        for ep_idx in progress:
            episode = source.meta.episodes[ep_idx]
            start = int(episode["dataset_from_index"])
            stop = int(episode["dataset_to_index"])

            kept_frames = 0
            for frame_idx in range(start, stop):
                item = source[frame_idx]
                output.add_frame(build_frame(item, output_features, args))
                kept_frames += 1

            if kept_frames == 0:
                raise RuntimeError(f"Episode {ep_idx} produced no frames.")

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
