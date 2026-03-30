from __future__ import annotations

import argparse

from .config import VisualizeDataConfig, default_config
from .dataset import create_all_episode_dataloaders, create_dataloader, load_dataset, save_stats_json
from .env_runner import close_env, create_env, replay_all_episodes, replay_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Визуализация записанных эпизодов SO-101.")
    parser.add_argument(
        "--episode",
        default="all",
        help="Индекс эпизода (0, 1, 2, ...) или 'all' для последовательного проигрывания всех эпизодов.",
    )
    parser.add_argument(
        "--save-stats",
        action="store_true",
        help="Сохранить stats.json после запуска.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> VisualizeDataConfig:
    config = default_config()
    config.save_stats = args.save_stats

    if args.episode == "all":
        config.episode = "all"
        return config

    try:
        config.episode = int(args.episode)
    except ValueError as exc:
        raise SystemExit("--episode должен быть целым числом или 'all'.") from exc

    if config.episode < 0:
        raise SystemExit("--episode должен быть неотрицательным.")

    return config


def main() -> None:
    args = parse_args()
    config = build_config(args)
    dataset = load_dataset(config)
    env = create_env(config)

    try:
        if config.episode == "all":
            episode_loaders = create_all_episode_dataloaders(dataset)
            replay_all_episodes(config, env, episode_loaders)
        else:
            dataloader, episode_sampler = create_dataloader(dataset, config.episode)
            replay_episode(config, env, dataloader, episode_sampler)
    finally:
        close_env(env)

    if config.save_stats:
        save_stats_json(dataset)


if __name__ == "__main__":
    main()
