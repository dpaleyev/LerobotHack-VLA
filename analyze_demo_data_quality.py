#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze demonstration quality and build a Markdown report.")
    parser.add_argument("--dataset-root", type=Path, default=Path("./demo_data2"))
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/analysis/demo_data2_quality"))
    return parser.parse_args()


def robust_positive_z(values: pd.Series, higher_is_worse: bool = True) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    if not higher_is_worse:
        arr = -arr
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    q25, q75 = np.percentile(arr, [25, 75])
    iqr_scale = (q75 - q25) / 1.349 if q75 > q25 else 0.0
    std_scale = arr.std()
    scale = max(1.4826 * mad, iqr_scale, std_scale, 1.0)
    z = (arr - median) / scale
    return pd.Series(np.clip(np.maximum(z, 0.0), 0.0, 5.0), index=values.index)


def compute_episode_metrics(parquet_path: Path) -> dict[str, float | int]:
    df = pd.read_parquet(
        parquet_path,
        columns=["observation.state", "action", "obj_init", "episode_index", "timestamp"],
    )

    ee = np.stack(df["observation.state"].to_numpy())
    action = np.stack(df["action"].to_numpy())
    obj_init = np.stack(df["obj_init"].to_numpy())

    ee_xyz = ee[:, :3]
    cube_xyz = obj_init[0, :3]
    plate_xyz = obj_init[0, 7:10]

    step_vec = np.diff(ee_xyz, axis=0) if len(ee_xyz) > 1 else np.empty((0, 3))
    step_xyz = np.linalg.norm(step_vec, axis=1) if len(ee_xyz) > 1 else np.array([])
    action_delta = np.linalg.norm(np.diff(action, axis=0), axis=1) if len(action) > 1 else np.array([])
    action_jerk = np.linalg.norm(np.diff(action, n=2, axis=0), axis=1) if len(action) > 2 else np.array([])
    cube_dist = np.linalg.norm(ee_xyz - cube_xyz, axis=1)
    closest_cube_dist = float(cube_dist.min())
    near_cube_threshold = max(closest_cube_dist + 0.015, 0.04)
    near_cube = cube_dist <= near_cube_threshold
    near_cube_entries = (
        int(near_cube[0]) + int(np.sum((~near_cube[:-1]) & near_cube[1:]))
        if len(near_cube)
        else 0
    )

    gripper_closed = action[:, 5] > 0.5
    close_events = (
        np.flatnonzero((~gripper_closed[:-1]) & (gripper_closed[1:])) + 1
        if len(action) > 1
        else np.array([], dtype=int)
    )
    open_events = (
        np.flatnonzero((gripper_closed[:-1]) & (~gripper_closed[1:])) + 1
        if len(action) > 1
        else np.array([], dtype=int)
    )
    first_close_idx = int(close_events[0]) if len(close_events) else -1
    close_near_cube = int(np.sum(cube_dist[close_events] <= near_cube_threshold)) if len(close_events) else 0
    far_close_events = int(len(close_events) - close_near_cube)
    regrasp_cycles = max(close_near_cube - 1, 0)

    turn_cos = np.array([])
    turn_angles_deg = np.array([])
    if len(step_vec) > 1:
        norms = np.linalg.norm(step_vec, axis=1)
        valid = (norms[:-1] > 1e-6) & (norms[1:] > 1e-6)
        if valid.any():
            u = step_vec[:-1][valid] / norms[:-1][valid, None]
            v = step_vec[1:][valid] / norms[1:][valid, None]
            turn_cos = np.clip(np.sum(u * v, axis=1), -1.0, 1.0)
            turn_angles_deg = np.degrees(np.arccos(turn_cos))

    path_len = float(step_xyz.sum())
    direct_dist = float(np.linalg.norm(ee_xyz[-1] - ee_xyz[0])) if len(ee_xyz) else 0.0

    return {
        "episode": int(df["episode_index"].iloc[0]),
        "length": int(len(df)),
        "duration_s": float(df["timestamp"].iloc[-1]),
        "ee_path_len": path_len,
        "ee_direct_dist": direct_dist,
        "path_efficiency": direct_dist / max(path_len, 1e-8),
        "idle_ratio": float((step_xyz < 0.002).mean()) if len(step_xyz) else 0.0,
        "ee_speed_mean": float(step_xyz.mean()) if len(step_xyz) else 0.0,
        "ee_speed_std": float(step_xyz.std()) if len(step_xyz) else 0.0,
        "turn_angle_mean_deg": float(turn_angles_deg.mean()) if len(turn_angles_deg) else 0.0,
        "turn_angle_p95_deg": float(np.percentile(turn_angles_deg, 95)) if len(turn_angles_deg) else 0.0,
        "reversal_ratio": float((turn_cos < 0.0).mean()) if len(turn_cos) else 0.0,
        "action_delta_mean": float(action_delta.mean()) if len(action_delta) else 0.0,
        "action_delta_p95": float(np.percentile(action_delta, 95)) if len(action_delta) else 0.0,
        "action_jerk_mean": float(action_jerk.mean()) if len(action_jerk) else 0.0,
        "action_jerk_p95": float(np.percentile(action_jerk, 95)) if len(action_jerk) else 0.0,
        "gripper_toggles": int(np.sum(gripper_closed[1:] != gripper_closed[:-1])) if len(action) > 1 else 0,
        "close_events": int(len(close_events)),
        "open_events": int(len(open_events)),
        "close_near_cube": close_near_cube,
        "far_close_events": far_close_events,
        "regrasp_cycles": regrasp_cycles,
        "near_cube_entries": near_cube_entries,
        "near_cube_ratio": float(near_cube.mean()) if len(near_cube) else 0.0,
        "first_close_idx": first_close_idx,
        "first_close_progress": first_close_idx / max(len(df) - 1, 1) if first_close_idx >= 0 else 1.0,
        "first_close_dist": float(cube_dist[first_close_idx]) if first_close_idx >= 0 else closest_cube_dist,
        "closest_cube_dist": closest_cube_dist,
        "final_ee_to_plate_dist": float(np.linalg.norm(ee_xyz[-1] - plate_xyz)),
        "cube_plate_xy_span": float(np.linalg.norm(cube_xyz[:2] - plate_xyz[:2])),
    }


def add_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    scored = metrics.copy()
    scored["z_length"] = robust_positive_z(scored["length"])
    scored["z_idle_ratio"] = robust_positive_z(scored["idle_ratio"])
    scored["z_toggles"] = robust_positive_z(scored["gripper_toggles"])
    scored["z_regrasp_cycles"] = robust_positive_z(scored["regrasp_cycles"])
    scored["z_far_close_events"] = robust_positive_z(scored["far_close_events"])
    scored["z_near_cube_entries"] = robust_positive_z(scored["near_cube_entries"])
    scored["z_first_close_dist"] = robust_positive_z(scored["first_close_dist"])
    scored["z_first_close_progress"] = robust_positive_z(scored["first_close_progress"])
    scored["z_action_jerk_p95"] = robust_positive_z(scored["action_jerk_p95"])
    scored["z_turn_angle_mean_deg"] = robust_positive_z(scored["turn_angle_mean_deg"])
    scored["z_low_efficiency"] = robust_positive_z(scored["path_efficiency"], higher_is_worse=False)
    scored["badness_score"] = (
        1.3 * scored["z_length"]
        + 1.1 * scored["z_idle_ratio"]
        + 1.0 * scored["z_toggles"]
        + 1.8 * scored["z_regrasp_cycles"]
        + 1.2 * scored["z_far_close_events"]
        + 1.0 * scored["z_near_cube_entries"]
        + 1.5 * scored["z_first_close_dist"]
        + 1.0 * scored["z_first_close_progress"]
        + 0.8 * scored["z_action_jerk_p95"]
        + 0.9 * scored["z_turn_angle_mean_deg"]
        + 0.8 * scored["z_low_efficiency"]
    )
    return scored


def add_flags(metrics: pd.DataFrame) -> pd.DataFrame:
    flagged = metrics.copy()
    p95_length = flagged["length"].quantile(0.95)
    p95_idle = flagged["idle_ratio"].quantile(0.95)
    p95_toggles = flagged["gripper_toggles"].quantile(0.95)
    p95_regrasp = max(1, int(np.ceil(flagged["regrasp_cycles"].quantile(0.95))))
    p90_far_close = max(2, int(np.ceil(flagged["far_close_events"].quantile(0.90))))
    p95_near_cube_entries = max(2, int(np.ceil(flagged["near_cube_entries"].quantile(0.95))))
    p90_first_close = flagged["first_close_dist"].quantile(0.90)
    p90_turn_angle = flagged["turn_angle_mean_deg"].quantile(0.90)
    p10_efficiency = flagged["path_efficiency"].quantile(0.10)
    p95_badness = flagged["badness_score"].quantile(0.95)
    p90_badness = flagged["badness_score"].quantile(0.90)

    flagged["flag_long_episode"] = flagged["length"] >= p95_length
    flagged["flag_idle"] = flagged["idle_ratio"] >= p95_idle
    flagged["flag_gripper_toggle"] = flagged["gripper_toggles"] >= p95_toggles
    flagged["flag_regrasp"] = flagged["regrasp_cycles"] >= p95_regrasp
    flagged["flag_far_close"] = flagged["far_close_events"] >= p90_far_close
    flagged["flag_many_reentries"] = flagged["near_cube_entries"] >= p95_near_cube_entries
    flagged["flag_far_first_close"] = flagged["first_close_dist"] >= p90_first_close
    flagged["flag_turning"] = flagged["turn_angle_mean_deg"] >= p90_turn_angle
    flagged["flag_low_efficiency"] = flagged["path_efficiency"] <= p10_efficiency
    flagged["flag_high_badness"] = flagged["badness_score"] >= p90_badness

    hard_drop = (
        (flagged["badness_score"] >= p95_badness)
        | (
            flagged[
                [
                    "flag_long_episode",
                    "flag_idle",
                    "flag_gripper_toggle",
                    "flag_regrasp",
                    "flag_far_close",
                    "flag_many_reentries",
                    "flag_far_first_close",
                    "flag_turning",
                    "flag_low_efficiency",
                ]
            ].sum(axis=1)
            >= 2
        )
    )
    review = (~hard_drop) & (flagged["badness_score"] >= p90_badness)
    flagged["recommendation"] = np.where(hard_drop, "hard_drop_candidate", np.where(review, "review", "keep"))
    return flagged


def save_plots(metrics: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}

    plt.figure(figsize=(8, 4.8))
    plt.hist(metrics["length"], bins=20, color="#4C78A8", edgecolor="black")
    plt.axvline(metrics["length"].quantile(0.90), color="orange", linestyle="--", label="p90")
    plt.axvline(metrics["length"].quantile(0.95), color="red", linestyle="--", label="p95")
    plt.title("Распределение длины эпизодов")
    plt.xlabel("Кадров в эпизоде")
    plt.ylabel("Число эпизодов")
    plt.legend()
    path = plot_dir / "episode_length_hist.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plots["episode_length_hist"] = str(path.relative_to(output_dir))

    plt.figure(figsize=(7.2, 5.4))
    scatter = plt.scatter(
        metrics["regrasp_cycles"],
        metrics["length"],
        c=metrics["badness_score"],
        cmap="viridis",
        s=28,
        alpha=0.9,
    )
    for _, row in metrics.nlargest(8, "badness_score").iterrows():
        plt.annotate(str(int(row["episode"])), (row["regrasp_cycles"], row["length"]), fontsize=8)
    plt.colorbar(scatter, label="Badness score")
    plt.title("Повторные попытки захвата vs длина эпизода")
    plt.xlabel("Повторные захваты рядом с кубиком")
    plt.ylabel("Кадров в эпизоде")
    path = plot_dir / "regrasp_vs_length.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plots["regrasp_vs_length"] = str(path.relative_to(output_dir))

    plt.figure(figsize=(7.2, 5.4))
    scatter = plt.scatter(
        metrics["turn_angle_mean_deg"],
        metrics["path_efficiency"],
        c=metrics["badness_score"],
        cmap="magma",
        s=28,
        alpha=0.9,
    )
    for _, row in metrics.nlargest(8, "badness_score").iterrows():
        plt.annotate(str(int(row["episode"])), (row["turn_angle_mean_deg"], row["path_efficiency"]), fontsize=8)
    plt.colorbar(scatter, label="Badness score")
    plt.title("Гладкость траектории vs эффективность пути")
    plt.xlabel("Средний угол поворота траектории (градусы)")
    plt.ylabel("Path efficiency")
    path = plot_dir / "smoothness_vs_efficiency.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plots["smoothness_vs_efficiency"] = str(path.relative_to(output_dir))

    plt.figure(figsize=(7.2, 5.4))
    scatter = plt.scatter(
        metrics["far_close_events"],
        metrics["near_cube_entries"],
        c=metrics["badness_score"],
        cmap="cividis",
        s=28,
        alpha=0.9,
    )
    for _, row in metrics.nlargest(8, "badness_score").iterrows():
        plt.annotate(str(int(row["episode"])), (row["far_close_events"], row["near_cube_entries"]), fontsize=8)
    plt.colorbar(scatter, label="Badness score")
    plt.title("Ранние закрытия и повторные заходы к кубику")
    plt.xlabel("Закрытия далеко от кубика")
    plt.ylabel("Число заходов в зону кубика")
    path = plot_dir / "far_close_vs_reentries.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plots["far_close_vs_reentries"] = str(path.relative_to(output_dir))

    plt.figure(figsize=(8, 5.2))
    top = metrics.nlargest(20, "badness_score").sort_values("badness_score")
    plt.barh(top["episode"].astype(str), top["badness_score"], color="#E45756")
    plt.title("Топ-20 самых подозрительных эпизодов")
    plt.xlabel("Badness score")
    plt.ylabel("Эпизод")
    path = plot_dir / "top_badness.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plots["top_badness"] = str(path.relative_to(output_dir))
    return plots


def frame_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    table = df.loc[:, columns].copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: f"{x:.3f}")
    headers = [str(col) for col in table.columns]
    rows = [[str(value) for value in row] for row in table.itertuples(index=False, name=None)]
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    markdown.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(markdown)


def image_block(path: str, alt: str, width: int = 900) -> str:
    return f'<img src="./{path}" alt="{alt}" width="{width}">'


def frame_to_html(df: pd.DataFrame, columns: list[str]) -> str:
    table = df.loc[:, columns].copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: f"{x:.3f}")
    return table.to_html(index=False, escape=False, border=0)


def build_report(metrics: pd.DataFrame, info: dict, plots: dict[str, str]) -> str:
    hard_drop = metrics[metrics["recommendation"] == "hard_drop_candidate"].sort_values("badness_score", ascending=False)
    review = metrics[metrics["recommendation"] == "review"].sort_values("badness_score", ascending=False)
    multi_attempt = metrics[metrics["regrasp_cycles"] >= 1].sort_values(
        ["regrasp_cycles", "badness_score"], ascending=[False, False]
    )
    criteria = pd.DataFrame(
        [
            {
                "criterion": "Длина эпизода",
                "meaning": "длинные серии обычно означают hesitation или лишние попытки",
                "median": metrics["length"].median(),
                "p90": metrics["length"].quantile(0.90),
                "worst": metrics["length"].max(),
            },
            {
                "criterion": "Повторные заходы к кубику",
                "meaning": "если робот уходит и снова возвращается к кубику, траектория нестабильна",
                "median": metrics["near_cube_entries"].median(),
                "p90": metrics["near_cube_entries"].quantile(0.90),
                "worst": metrics["near_cube_entries"].max(),
            },
            {
                "criterion": "Ранние закрытия gripper",
                "meaning": "закрытие далеко от кубика похоже на неудачную попытку схвата",
                "median": metrics["far_close_events"].median(),
                "p90": metrics["far_close_events"].quantile(0.90),
                "worst": metrics["far_close_events"].max(),
            },
            {
                "criterion": "Ломаность траектории",
                "meaning": "большой средний угол поворота значит, что рука часто меняла направление",
                "median": metrics["turn_angle_mean_deg"].median(),
                "p90": metrics["turn_angle_mean_deg"].quantile(0.90),
                "worst": metrics["turn_angle_mean_deg"].max(),
            },
            {
                "criterion": "Эффективность пути",
                "meaning": "чем ниже path efficiency, тем больше блуждания вместо прямого подхода",
                "median": metrics["path_efficiency"].median(),
                "p10": metrics["path_efficiency"].quantile(0.10),
                "worst": metrics["path_efficiency"].min(),
            },
        ]
    )
    criteria["median"] = criteria["median"].map(lambda x: f"{x:.2f}")
    criteria["worst"] = criteria["worst"].map(lambda x: f"{x:.2f}")
    criteria["p90_or_p10"] = [
        f"{metrics['length'].quantile(0.90):.2f}",
        f"{metrics['near_cube_entries'].quantile(0.90):.2f}",
        f"{metrics['far_close_events'].quantile(0.90):.2f}",
        f"{metrics['turn_angle_mean_deg'].quantile(0.90):.2f}",
        f"{metrics['path_efficiency'].quantile(0.10):.2f}",
    ]
    criteria = criteria[["criterion", "meaning", "median", "p90_or_p10", "worst"]]
    top_bad = metrics.sort_values("badness_score", ascending=False).head(10)
    lines = [
        "# Отчет по качеству демонстраций",
        "",
        f"Dataset root: `{info['dataset_root']}`",
        "",
        "## Коротко",
        "",
        f"В `{Path(info['dataset_root']).name}` всего **{info['total_episodes']}** эпизодов и **{info['total_frames']}** кадров при **{info['fps']} FPS**.",
        f"По длине датасет уже выглядит неоднородным: медиана эпизода **{metrics['length'].median():.0f}** кадров, а верхние 10% начинаются примерно с **{metrics['length'].quantile(0.90):.0f}** кадров.",
        f"Явный повторный захват рядом с кубиком встретился только в **{len(multi_attempt)}** эпизоде, зато паттерн \"подошел, отошел, снова подошел\" встречается заметно чаще.",
        "",
        "## История датасета",
        "",
        "Если смотреть на датасет не как на набор файлов, а как на поведение робота, то шум здесь проявляется в пяти местах.",
        "Сначала видно, что часть эпизодов просто слишком длинная. Обычно это не про сложность задачи, а про сомнения: робот долго подруливает, медлит или несколько раз передумывает.",
        "Потом видно повторные заходы к кубику. Это более мягкий, но очень важный сигнал: даже если явного повторного схвата нет, робот может несколько раз возвращаться в зону кубика, и это уже делает демонстрацию менее чистой.",
        "Третий сигнал это ранние закрытия gripper. Когда gripper закрывается, пока рука еще далеко от кубика, это похоже на неудачную попытку или на плохо синхронизированный motion pattern.",
        "Четвертый сигнал это ломаность траектории. Гладкая демонстрация обычно идет почти напрямую, а шумная постоянно меняет направление.",
        "Пятый сигнал это эффективность пути. Она хорошо дополняет гладкость: траектория может быть не очень дерганой, но все равно длинной и блуждающей.",
        "",
        "## Сравнение 5 критериев",
        "",
        frame_to_markdown(criteria, ["criterion", "meaning", "median", "p90_or_p10", "worst"]),
        "",
        "## Что важнее всего",
        "",
        f"Сильнее всего выбиваются не эпизоды с буквальным многократным схватом, а эпизоды, где одновременно длинная серия, много повторных заходов к кубику и очень плохая эффективность пути.",
        f"Лучшие кандидаты на выброс или ручную проверку это эпизоды `{', '.join(str(int(x)) for x in top_bad['episode'].head(6))}`.",
        f"Сейчас по эвристике получилось **{len(hard_drop)}** `hard_drop_candidate` и **{len(review)}** `review`.",
        "",
        "## Графики",
        "",
        "Сначала распределение длины. Здесь хорошо видно, где начинается длинный хвост эпизодов.",
        "",
        image_block(plots["episode_length_hist"], "Episode length histogram"),
        "",
        "Дальше связь между длиной и повторными попытками. Этот график полезен именно для твоей гипотезы про повторные захваты.",
        "",
        image_block(plots["regrasp_vs_length"], "Regrasp vs length"),
        "",
        "Здесь сравнение гладкости и эффективности пути. Хорошие траектории обычно лежат ниже по углам поворота и выше по эффективности.",
        "",
        image_block(plots["smoothness_vs_efficiency"], "Smoothness vs efficiency"),
        "",
        "А этот график показывает, где ранние закрытия gripper сочетаются с повторными заходами к кубику.",
        "",
        image_block(plots["far_close_vs_reentries"], "Far closes vs reentries"),
        "",
        "В конце сводный рейтинг самых шумных эпизодов.",
        "",
        image_block(plots["top_badness"], "Top badness episodes"),
        "",
        "## Самые подозрительные эпизоды",
        "",
        frame_to_markdown(
            top_bad,
            [
                "episode",
                "badness_score",
                "length",
                "far_close_events",
                "near_cube_entries",
                "turn_angle_mean_deg",
                "path_efficiency",
            ],
        ),
        "",
        "## Вывод",
        "",
        "Если говорить совсем просто, проблема этого датасета не в том, что робот много раз физически перехватывает кубик. Проблема скорее в том, что часть траекторий слишком длинные, с возвратами к кубику и с неэффективным, ломаным движением.",
        "То есть для фильтрации я бы в первую очередь смотрел на комбинацию `length`, `near_cube_entries`, `far_close_events`, `turn_angle_mean_deg` и `path_efficiency`.",
        "",
    ]
    return "\n".join(lines)


def build_html_report(metrics: pd.DataFrame, info: dict, plots: dict[str, str]) -> str:
    hard_drop = metrics[metrics["recommendation"] == "hard_drop_candidate"].sort_values("badness_score", ascending=False)
    review = metrics[metrics["recommendation"] == "review"].sort_values("badness_score", ascending=False)
    multi_attempt = metrics[metrics["regrasp_cycles"] >= 1].sort_values(
        ["regrasp_cycles", "badness_score"], ascending=[False, False]
    )
    criteria = pd.DataFrame(
        [
            {
                "criterion": "Длина эпизода",
                "meaning": "длинные серии обычно означают hesitation или лишние попытки",
                "median": metrics["length"].median(),
                "p90_or_p10": metrics["length"].quantile(0.90),
                "worst": metrics["length"].max(),
            },
            {
                "criterion": "Повторные заходы к кубику",
                "meaning": "если робот уходит и снова возвращается к кубику, траектория нестабильна",
                "median": metrics["near_cube_entries"].median(),
                "p90_or_p10": metrics["near_cube_entries"].quantile(0.90),
                "worst": metrics["near_cube_entries"].max(),
            },
            {
                "criterion": "Ранние закрытия gripper",
                "meaning": "закрытие далеко от кубика похоже на неудачную попытку схвата",
                "median": metrics["far_close_events"].median(),
                "p90_or_p10": metrics["far_close_events"].quantile(0.90),
                "worst": metrics["far_close_events"].max(),
            },
            {
                "criterion": "Ломаность траектории",
                "meaning": "большой средний угол поворота значит, что рука часто меняла направление",
                "median": metrics["turn_angle_mean_deg"].median(),
                "p90_or_p10": metrics["turn_angle_mean_deg"].quantile(0.90),
                "worst": metrics["turn_angle_mean_deg"].max(),
            },
            {
                "criterion": "Эффективность пути",
                "meaning": "чем ниже path efficiency, тем больше блуждания вместо прямого подхода",
                "median": metrics["path_efficiency"].median(),
                "p90_or_p10": metrics["path_efficiency"].quantile(0.10),
                "worst": metrics["path_efficiency"].min(),
            },
        ]
    )
    top_bad = metrics.sort_values("badness_score", ascending=False).head(10)
    image_paths = [
        ("Сначала распределение длины. Здесь хорошо видно, где начинается длинный хвост эпизодов.", plots["episode_length_hist"], "Episode length histogram"),
        ("Дальше связь между длиной и повторными попытками. Этот график полезен именно для твоей гипотезы про повторные захваты.", plots["regrasp_vs_length"], "Regrasp vs length"),
        ("Здесь сравнение гладкости и эффективности пути. Хорошие траектории обычно лежат ниже по углам поворота и выше по эффективности.", plots["smoothness_vs_efficiency"], "Smoothness vs efficiency"),
        ("А этот график показывает, где ранние закрытия gripper сочетаются с повторными заходами к кубику.", plots["far_close_vs_reentries"], "Far closes vs reentries"),
        ("В конце сводный рейтинг самых шумных эпизодов.", plots["top_badness"], "Top badness episodes"),
    ]
    image_html = "\n".join(
        f"<p>{text}</p>\n<img src=\"./{path}\" alt=\"{alt}\">" for text, path, alt in image_paths
    )
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Отчет по качеству демонстраций</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      line-height: 1.5;
      margin: 32px auto;
      max-width: 1100px;
      padding: 0 20px 40px;
      color: #222;
    }}
    h1, h2 {{ margin-top: 28px; }}
    p {{ margin: 10px 0; }}
    code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }}
    img {{ max-width: 100%; display: block; margin: 14px 0 28px; border: 1px solid #ddd; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 28px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f7f7f7; }}
  </style>
</head>
<body>
  <h1>Отчет по качеству демонстраций</h1>
  <p>Dataset root: <code>{info['dataset_root']}</code></p>

  <h2>Коротко</h2>
  <p>В <code>{Path(info['dataset_root']).name}</code> всего <strong>{info['total_episodes']}</strong> эпизодов и <strong>{info['total_frames']}</strong> кадров при <strong>{info['fps']} FPS</strong>.</p>
  <p>По длине датасет уже выглядит неоднородным: медиана эпизода <strong>{metrics['length'].median():.0f}</strong> кадров, а верхние 10% начинаются примерно с <strong>{metrics['length'].quantile(0.90):.0f}</strong> кадров.</p>
  <p>Явный повторный захват рядом с кубиком встретился только в <strong>{len(multi_attempt)}</strong> эпизоде, зато паттерн "подошел, отошел, снова подошел" встречается заметно чаще.</p>

  <h2>История датасета</h2>
  <p>Если смотреть на датасет не как на набор файлов, а как на поведение робота, то шум здесь проявляется в пяти местах.</p>
  <p>Сначала видно, что часть эпизодов просто слишком длинная. Обычно это не про сложность задачи, а про сомнения: робот долго подруливает, медлит или несколько раз передумывает.</p>
  <p>Потом видно повторные заходы к кубику. Это более мягкий, но очень важный сигнал: даже если явного повторного схвата нет, робот может несколько раз возвращаться в зону кубика, и это уже делает демонстрацию менее чистой.</p>
  <p>Третий сигнал это ранние закрытия gripper. Когда gripper закрывается, пока рука еще далеко от кубика, это похоже на неудачную попытку или на плохо синхронизированный motion pattern.</p>
  <p>Четвертый сигнал это ломаность траектории. Гладкая демонстрация обычно идет почти напрямую, а шумная постоянно меняет направление.</p>
  <p>Пятый сигнал это эффективность пути. Она хорошо дополняет гладкость: траектория может быть не очень дерганой, но все равно длинной и блуждающей.</p>

  <h2>Сравнение 5 критериев</h2>
  {frame_to_html(criteria, ["criterion", "meaning", "median", "p90_or_p10", "worst"])}

  <h2>Что важнее всего</h2>
  <p>Сильнее всего выбиваются не эпизоды с буквальным многократным схватом, а эпизоды, где одновременно длинная серия, много повторных заходов к кубику и очень плохая эффективность пути.</p>
  <p>Лучшие кандидаты на выброс или ручную проверку это эпизоды <code>{', '.join(str(int(x)) for x in top_bad['episode'].head(6))}</code>.</p>
  <p>Сейчас по эвристике получилось <strong>{len(hard_drop)}</strong> <code>hard_drop_candidate</code> и <strong>{len(review)}</strong> <code>review</code>.</p>

  <h2>Графики</h2>
  {image_html}

  <h2>Самые подозрительные эпизоды</h2>
  {frame_to_html(top_bad, ["episode", "badness_score", "length", "far_close_events", "near_cube_entries", "turn_angle_mean_deg", "path_efficiency"])}

  <h2>Вывод</h2>
  <p>Если говорить совсем просто, проблема этого датасета не в том, что робот много раз физически перехватывает кубик. Проблема скорее в том, что часть траекторий слишком длинные, с возвратами к кубику и с неэффективным, ломаным движением.</p>
  <p>То есть для фильтрации я бы в первую очередь смотрел на комбинацию <code>length</code>, <code>near_cube_entries</code>, <code>far_close_events</code>, <code>turn_angle_mean_deg</code> и <code>path_efficiency</code>.</p>
</body>
</html>
"""


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    parquet_files = sorted((args.dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    metrics = pd.DataFrame([compute_episode_metrics(path) for path in parquet_files]).sort_values("episode")
    metrics = add_scores(metrics)
    metrics = add_flags(metrics)

    metrics_csv = args.output_dir / "episode_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    plots = save_plots(metrics, args.output_dir)
    report = build_report(
        metrics=metrics,
        info={
            "dataset_root": str(args.dataset_root),
            "total_episodes": info["total_episodes"],
            "total_frames": info["total_frames"],
            "fps": info["fps"],
        },
        plots=plots,
    )
    report_path = args.output_dir / "report.md"
    report_path.write_text(report + "\n")
    html_report_path = args.output_dir / "report.html"
    html_report_path.write_text(build_html_report(metrics=metrics, info={
        "dataset_root": str(args.dataset_root),
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "fps": info["fps"],
    }, plots=plots) + "\n")

    suspicious_path = args.output_dir / "suspicious_episodes.csv"
    metrics[metrics["recommendation"] != "keep"].sort_values("badness_score", ascending=False).to_csv(
        suspicious_path, index=False
    )

    print(f"Saved report: {report_path}")
    print(f"Saved html report: {html_report_path}")
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved suspicious list: {suspicious_path}")


if __name__ == "__main__":
    main()
