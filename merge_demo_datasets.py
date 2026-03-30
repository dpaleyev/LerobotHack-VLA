#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple demo datasets into one draft dataset.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input dataset roots in the desired merge order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output merged dataset root.",
    )
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
    input_roots = [Path(path) for path in args.inputs]
    output_root = args.output

    if output_root.exists():
        raise SystemExit(f"Output already exists: {output_root}")

    infos = [json.loads((root / "meta" / "info.json").read_text()) for root in input_roots]
    reference = infos[0]
    for root, info in zip(input_roots, infos, strict=True):
        if info["features"] != reference["features"]:
            raise SystemExit(f"Feature mismatch in {root}")
        if info["fps"] != reference["fps"]:
            raise SystemExit(f"FPS mismatch in {root}")
        if info["robot_type"] != reference["robot_type"]:
            raise SystemExit(f"Robot type mismatch in {root}")

    (output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=False)
    (output_root / "meta").mkdir(parents=True, exist_ok=False)

    merged_episodes: list[dict] = []
    merged_episode_stats: list[dict] = []
    merge_manifest: list[dict] = []

    next_episode_index = 0
    next_global_index = 0

    for root in input_roots:
        episode_rows = read_jsonl(root / "meta" / "episodes.jsonl")
        episode_stats_rows = read_jsonl(root / "meta" / "episodes_stats.jsonl")
        stats_by_episode = {row["episode_index"]: row for row in episode_stats_rows}
        parquet_files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))

        for episode_row, parquet_path in zip(episode_rows, parquet_files, strict=True):
            table = pq.read_table(parquet_path)
            old_episode_index = int(episode_row["episode_index"])
            length = table.num_rows

            table = replace_column(table, "episode_index", [next_episode_index] * length)
            table = replace_column(table, "frame_index", range(length))
            table = replace_column(table, "index", range(next_global_index, next_global_index + length))

            target_parquet = output_root / "data" / "chunk-000" / f"episode_{next_episode_index:06d}.parquet"
            pq.write_table(table, target_parquet)

            merged_episodes.append(
                {
                    "episode_index": next_episode_index,
                    "tasks": episode_row["tasks"],
                    "length": length,
                }
            )

            stats_row = dict(stats_by_episode[old_episode_index])
            stats_row["episode_index"] = next_episode_index
            if "stats" in stats_row:
                if "episode_index" in stats_row["stats"]:
                    stats_row["stats"]["episode_index"]["min"] = [next_episode_index]
                    stats_row["stats"]["episode_index"]["max"] = [next_episode_index]
                    stats_row["stats"]["episode_index"]["mean"] = [float(next_episode_index)]
                    stats_row["stats"]["episode_index"]["std"] = [0.0]
                if "index" in stats_row["stats"]:
                    stats_row["stats"]["index"]["min"] = [next_global_index]
                    stats_row["stats"]["index"]["max"] = [next_global_index + length - 1]
                    stats_row["stats"]["index"]["mean"] = [next_global_index + (length - 1) / 2]
            merged_episode_stats.append(stats_row)

            merge_manifest.append(
                {
                    "source_dataset": root.name,
                    "source_episode_index": old_episode_index,
                    "merged_episode_index": next_episode_index,
                    "length": length,
                    "global_index_start": next_global_index,
                    "global_index_end": next_global_index + length - 1,
                }
            )

            next_global_index += length
            next_episode_index += 1

    tasks = read_jsonl(input_roots[0] / "meta" / "tasks.jsonl")
    write_jsonl(output_root / "meta" / "tasks.jsonl", tasks)
    write_jsonl(output_root / "meta" / "episodes.jsonl", merged_episodes)
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", merged_episode_stats)
    write_jsonl(output_root / "meta" / "merge_manifest.jsonl", merge_manifest)

    merged_info = dict(reference)
    merged_info["total_episodes"] = len(merged_episodes)
    merged_info["total_frames"] = next_global_index
    merged_info["total_tasks"] = len(tasks)
    merged_info["total_chunks"] = 1
    merged_info["splits"] = {"train": f"0:{len(merged_episodes)}"}
    (output_root / "meta" / "info.json").write_text(json.dumps(merged_info, indent=4) + "\n")

    print(f"Merged datasets into: {output_root}")
    print(f"Total episodes: {len(merged_episodes)}")
    print(f"Total frames: {next_global_index}")


if __name__ == "__main__":
    main()
