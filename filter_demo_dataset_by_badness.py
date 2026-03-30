#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="Filter a demo dataset using badness_score.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Source merged dataset root.")
    parser.add_argument("--metrics-csv", type=Path, required=True, help="CSV with episode metrics and badness_score.")
    parser.add_argument("--max-badness", type=float, required=True, help="Keep episodes with badness_score <= this threshold.")
    parser.add_argument("--output-root", type=Path, required=True, help="Filtered dataset output root.")
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

    metrics = pd.read_csv(args.metrics_csv)
    keep = metrics.loc[metrics["badness_score"] <= args.max_badness].sort_values("episode")
    keep_episodes = keep["episode"].astype(int).tolist()
    keep_set = set(keep_episodes)

    source_info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    source_episodes = read_jsonl(args.dataset_root / "meta" / "episodes.jsonl")
    source_episode_stats = read_jsonl(args.dataset_root / "meta" / "episodes_stats.jsonl")
    source_stats_by_episode = {row["episode_index"]: row for row in source_episode_stats}
    source_tasks = read_jsonl(args.dataset_root / "meta" / "tasks.jsonl")

    (args.output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=False)
    (args.output_root / "meta").mkdir(parents=True, exist_ok=False)

    filtered_episodes: list[dict] = []
    filtered_episode_stats: list[dict] = []
    filter_manifest: list[dict] = []

    new_episode_index = 0
    new_global_index = 0

    for episode_row in source_episodes:
        old_episode_index = int(episode_row["episode_index"])
        if old_episode_index not in keep_set:
            continue

        parquet_path = args.dataset_root / "data" / "chunk-000" / f"episode_{old_episode_index:06d}.parquet"
        table = pq.read_table(parquet_path)
        length = table.num_rows

        table = replace_column(table, "episode_index", [new_episode_index] * length)
        table = replace_column(table, "frame_index", range(length))
        table = replace_column(table, "index", range(new_global_index, new_global_index + length))

        output_parquet = args.output_root / "data" / "chunk-000" / f"episode_{new_episode_index:06d}.parquet"
        pq.write_table(table, output_parquet)

        filtered_episodes.append(
            {
                "episode_index": new_episode_index,
                "tasks": episode_row["tasks"],
                "length": length,
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
            if "index" in stats_row["stats"]:
                stats_row["stats"]["index"]["min"] = [new_global_index]
                stats_row["stats"]["index"]["max"] = [new_global_index + length - 1]
                stats_row["stats"]["index"]["mean"] = [new_global_index + (length - 1) / 2]
        filtered_episode_stats.append(stats_row)

        row_metrics = keep.loc[keep["episode"] == old_episode_index].iloc[0]
        filter_manifest.append(
            {
                "source_episode_index": old_episode_index,
                "filtered_episode_index": new_episode_index,
                "length": length,
                "badness_score": float(row_metrics["badness_score"]),
                "global_index_start": new_global_index,
                "global_index_end": new_global_index + length - 1,
            }
        )

        new_global_index += length
        new_episode_index += 1

    write_jsonl(args.output_root / "meta" / "tasks.jsonl", source_tasks)
    write_jsonl(args.output_root / "meta" / "episodes.jsonl", filtered_episodes)
    write_jsonl(args.output_root / "meta" / "episodes_stats.jsonl", filtered_episode_stats)
    write_jsonl(args.output_root / "meta" / "filter_manifest.jsonl", filter_manifest)

    filtered_info = dict(source_info)
    filtered_info["total_episodes"] = len(filtered_episodes)
    filtered_info["total_frames"] = new_global_index
    filtered_info["splits"] = {"train": f"0:{len(filtered_episodes)}"}
    (args.output_root / "meta" / "info.json").write_text(json.dumps(filtered_info, indent=4) + "\n")

    keep.to_csv(args.output_root / "meta" / "kept_episode_metrics.csv", index=False)

    print(f"Filtered dataset saved to: {args.output_root}")
    print(f"Kept episodes: {len(filtered_episodes)}")
    print(f"Kept frames: {new_global_index}")


if __name__ == "__main__":
    main()
