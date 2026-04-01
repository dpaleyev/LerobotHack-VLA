#!/usr/bin/env python3
from __future__ import annotations

# DEPRECATED: legacy numeric contract repair path.
# Canonical pipeline uses collect_data with joint .pos state/action.
# See DATASET_FORMAT_DEPRECATIONS.md.

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from collect_data.config import default_config
from lerobot.datasets.dataset_tools import write_info, write_stats
from mujoco_env.y_env import SimpleEnv

QUANTILES = (
    ("q01", 0.01),
    ("q10", 0.10),
    ("q50", 0.50),
    ("q90", 0.90),
    ("q99", 0.99),
)
GENERIC_ACTION_NAMES = ["action"]
GENERIC_STATE_NAMES = ["state"]


def parse_args() -> argparse.Namespace:
    cfg = default_config()
    parser = argparse.ArgumentParser(
        description="Fix dataset numeric contract while preserving video files."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["real_joint_deg", "sim_realfmt_passthrough"],
        required=True,
        help="How to interpret existing observation.state and action columns.",
    )
    parser.add_argument("--xml-path", default=cfg.xml_path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def replace_column(table: pa.Table, name: str, values: list) -> pa.Table:
    idx = table.schema.get_field_index(name)
    field = table.schema.field(idx)
    array = pa.array(values, type=field.type)
    return table.set_column(idx, field, array)


def vector_stats(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float32)
    stats: dict[str, list] = {
        "min": values.min(axis=0).astype(np.float32).tolist(),
        "max": values.max(axis=0).astype(np.float32).tolist(),
        "mean": values.mean(axis=0).astype(np.float32).tolist(),
        "std": values.std(axis=0).astype(np.float32).tolist(),
        "count": [int(values.shape[0])],
    }
    for key, q in QUANTILES:
        stats[key] = np.quantile(values, q, axis=0).astype(np.float32).tolist()
    return stats


def make_fk_env(xml_path: str) -> SimpleEnv:
    return SimpleEnv(
        xml_path=xml_path,
        action_type="joint_angle",
        state_type="joint_angle",
        seed=0,
        visible_window=False,
    )


def convert_real_home_numeric(state_deg: np.ndarray, action_deg: np.ndarray, xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    fk_env = make_fk_env(xml_path)
    state_out = np.empty((len(state_deg), 6), dtype=np.float32)
    action_out = np.deg2rad(action_deg).astype(np.float32)

    for idx, joint_deg in enumerate(tqdm(state_deg, desc="FK real-home", unit="frame")):
        arm_q_rad = np.deg2rad(joint_deg[:5]).astype(np.float32)
        fk_env.env.set_qpos_joints(fk_env.arm_joint_names, arm_q_rad)
        state_out[idx] = fk_env.get_ee_pose().astype(np.float32)

    return state_out, action_out


def prepare_numeric_columns(table: pa.Table, mode: str, xml_path: str) -> tuple[np.ndarray, np.ndarray]:
    columns = table.select(["observation.state", "action"]).to_pydict()
    state = np.asarray(columns["observation.state"], dtype=np.float32)
    action = np.asarray(columns["action"], dtype=np.float32)

    if mode == "real_joint_deg":
        return convert_real_home_numeric(state, action, xml_path)

    if mode == "sim_realfmt_passthrough":
        return state.astype(np.float32), action.astype(np.float32)

    raise ValueError(f"Unsupported mode: {mode}")


def rewrite_info(info: dict) -> dict:
    new_info = json.loads(json.dumps(info))
    new_info["features"]["action"]["names"] = GENERIC_ACTION_NAMES
    new_info["features"]["observation.state"]["names"] = GENERIC_STATE_NAMES
    return new_info


def rewrite_episode_stats(
    episodes_table: pa.Table,
    state_values: np.ndarray,
    action_values: np.ndarray,
) -> pa.Table:
    rows = episodes_table.to_pydict()
    starts = rows["dataset_from_index"]
    stops = rows["dataset_to_index"]

    action_stats_by_episode = [vector_stats(action_values[start:stop]) for start, stop in zip(starts, stops, strict=True)]
    state_stats_by_episode = [vector_stats(state_values[start:stop]) for start, stop in zip(starts, stops, strict=True)]

    for stat_key in ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]:
        episodes_table = replace_column(
            episodes_table,
            f"stats/action/{stat_key}",
            [item[stat_key] for item in action_stats_by_episode],
        )
        episodes_table = replace_column(
            episodes_table,
            f"stats/observation.state/{stat_key}",
            [item[stat_key] for item in state_stats_by_episode],
        )

    return episodes_table


def copy_static_layout(input_root: Path, output_root: Path) -> None:
    shutil.copytree(input_root / "videos", output_root / "videos")
    shutil.copy2(input_root / "meta" / "tasks.parquet", output_root / "meta" / "tasks.parquet")


def main() -> None:
    args = parse_args()

    if args.output_root.exists():
        raise SystemExit(f"Output already exists: {args.output_root}")

    info = load_json(args.input_root / "meta" / "info.json")
    stats = load_json(args.input_root / "meta" / "stats.json")

    (args.output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=False)
    (args.output_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=False)

    data_path = args.input_root / "data" / "chunk-000" / "file-000.parquet"
    episodes_path = args.input_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

    data_table = pq.read_table(data_path)
    episodes_table = pq.read_table(episodes_path)

    state_values, action_values = prepare_numeric_columns(data_table, args.mode, args.xml_path)
    data_table = replace_column(data_table, "observation.state", state_values.tolist())
    data_table = replace_column(data_table, "action", action_values.tolist())

    new_stats = json.loads(json.dumps(stats))
    new_stats["observation.state"] = vector_stats(state_values)
    new_stats["action"] = vector_stats(action_values)
    new_info = rewrite_info(info)
    new_episodes = rewrite_episode_stats(episodes_table, state_values, action_values)

    save_table(data_table, args.output_root / "data" / "chunk-000" / "file-000.parquet")
    save_table(new_episodes, args.output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    write_info(new_info, args.output_root)
    write_stats(new_stats, args.output_root)
    copy_static_layout(args.input_root, args.output_root)

    print(f"Fixed dataset saved to: {args.output_root}")
    print(f"Mode: {args.mode}")
    print(f"Frames: {len(state_values)}")


if __name__ == "__main__":
    main()
