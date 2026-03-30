#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="Trim the first N frames from every episode in a demo dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Source dataset root.")
    parser.add_argument("--trim-frames", type=int, required=True, help="Number of initial frames to drop from each episode.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output trimmed dataset root.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def replace_column(table: pa.Table, name: str, values: list | range) -> pa.Table:
    idx = table.schema.get_field_index(name)
    field = table.schema.field(idx)
    array = pa.array(values, type=field.type)
    return table.set_column(idx, field, array)


def main():
    args = parse_args()
    if args.output_root.exists():
        raise SystemExit(f"Output already exists: {args.output_root}")
    if args.trim_frames < 0:
        raise SystemExit("trim_frames must be non-negative")

    source_info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    source_episodes = read_jsonl(args.dataset_root / "meta" / "episodes.jsonl")
    source_episode_stats = read_jsonl(args.dataset_root / "meta" / "episodes_stats.jsonl")
    source_tasks = read_jsonl(args.dataset_root / "meta" / "tasks.jsonl")
    source_stats_by_episode = {row["episode_index"]: row for row in source_episode_stats}

    (args.output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=False)
    (args.output_root / "meta").mkdir(parents=True, exist_ok=False)

    trimmed_episodes: list[dict] = []
    trimmed_stats: list[dict] = []
    trim_manifest: list[dict] = []

    total_frames = 0

    for new_episode_index, episode_row in enumerate(source_episodes):
        old_episode_index = int(episode_row["episode_index"])
        parquet_path = args.dataset_root / "data" / "chunk-000" / f"episode_{old_episode_index:06d}.parquet"
        table = pq.read_table(parquet_path)
        if table.num_rows <= args.trim_frames:
            raise SystemExit(f"Episode {old_episode_index} too short for trimming: len={table.num_rows}")

        trimmed = table.slice(args.trim_frames)
        trimmed_length = trimmed.num_rows
        trimmed = replace_column(trimmed, "episode_index", [new_episode_index] * trimmed_length)
        trimmed = replace_column(trimmed, "frame_index", range(trimmed_length))
        trimmed = replace_column(trimmed, "index", range(total_frames, total_frames + trimmed_length))
        if "timestamp" in trimmed.column_names and trimmed_length:
            timestamp_idx = trimmed.schema.get_field_index("timestamp")
            timestamp_field = trimmed.schema.field(timestamp_idx)
            timestamps = trimmed.column(timestamp_idx).to_pylist()
            first_ts = float(timestamps[0])
            timestamps = [float(ts) - first_ts for ts in timestamps]
            trimmed = trimmed.set_column(timestamp_idx, timestamp_field, pa.array(timestamps, type=timestamp_field.type))

        output_parquet = args.output_root / "data" / "chunk-000" / f"episode_{new_episode_index:06d}.parquet"
        pq.write_table(trimmed, output_parquet)

        trimmed_episodes.append(
            {
                "episode_index": new_episode_index,
                "tasks": episode_row["tasks"],
                "length": int(trimmed_length),
            }
        )

        stats_row = dict(source_stats_by_episode[old_episode_index])
        stats_row["episode_index"] = new_episode_index
        if "stats" in stats_row:
            if "episode_index" in stats_row["stats"]:
                stats_row["stats"]["episode_index"]["min"] = [new_episode_index]
                stats_row["stats"]["episode_index"]["max"] = [new_episode_index]
                stats_row["stats"]["episode_index"]["mean"] = [float(new_episode_index)]
                stats_row["stats"]["episode_index"]["std"] = [0.0]
            if "frame_index" in stats_row["stats"]:
                stats_row["stats"]["frame_index"]["min"] = [0]
                stats_row["stats"]["frame_index"]["max"] = [trimmed_length - 1]
                stats_row["stats"]["frame_index"]["mean"] = [(trimmed_length - 1) / 2]
                stats_row["stats"]["frame_index"]["count"] = [trimmed_length]
            if "index" in stats_row["stats"]:
                stats_row["stats"]["index"]["min"] = [total_frames]
                stats_row["stats"]["index"]["max"] = [total_frames + trimmed_length - 1]
                stats_row["stats"]["index"]["mean"] = [total_frames + (trimmed_length - 1) / 2]
                stats_row["stats"]["index"]["count"] = [trimmed_length]
            if "timestamp" in stats_row["stats"]:
                old_max = float(stats_row["stats"]["timestamp"]["max"][0])
                old_count = int(stats_row["stats"]["timestamp"]["count"][0])
                dt = old_max / max(old_count - 1, 1)
                new_max = dt * max(trimmed_length - 1, 0)
                stats_row["stats"]["timestamp"]["min"] = [0.0]
                stats_row["stats"]["timestamp"]["max"] = [new_max]
                stats_row["stats"]["timestamp"]["mean"] = [new_max / 2]
                stats_row["stats"]["timestamp"]["count"] = [trimmed_length]
        trimmed_stats.append(stats_row)

        trim_manifest.append(
            {
                "source_episode_index": old_episode_index,
                "trimmed_episode_index": new_episode_index,
                "trim_frames": args.trim_frames,
                "source_length": int(table.num_rows),
                "trimmed_length": int(trimmed_length),
                "global_index_start": total_frames,
                "global_index_end": total_frames + trimmed_length - 1,
            }
        )

        total_frames += trimmed_length

    write_jsonl(args.output_root / "meta" / "tasks.jsonl", source_tasks)
    write_jsonl(args.output_root / "meta" / "episodes.jsonl", trimmed_episodes)
    write_jsonl(args.output_root / "meta" / "episodes_stats.jsonl", trimmed_stats)
    write_jsonl(args.output_root / "meta" / "trim_manifest.jsonl", trim_manifest)

    trimmed_info = dict(source_info)
    trimmed_info["total_episodes"] = len(trimmed_episodes)
    trimmed_info["total_frames"] = total_frames
    trimmed_info["splits"] = {"train": f"0:{len(trimmed_episodes)}"}
    (args.output_root / "meta" / "info.json").write_text(json.dumps(trimmed_info, indent=4) + "\n")

    print(f"Trimmed dataset saved to: {args.output_root}")
    print(f"Trimmed episodes: {len(trimmed_episodes)}")
    print(f"Trimmed frames: {total_frames}")


if __name__ == "__main__":
    main()
