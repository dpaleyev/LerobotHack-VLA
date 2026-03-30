from __future__ import annotations

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import serialize_dict, write_json

from .config import VisualizeDataConfig


class EpisodeSampler(torch.utils.data.Sampler):
    """Sampler for one selected episode."""

    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def load_dataset(config: VisualizeDataConfig) -> LeRobotDataset:
    dataset = LeRobotDataset(config.repo_name, root=str(config.root))
    print(f"Датасет загружен: {config.repo_name}")
    print(f"Корневая папка: {dataset.root}")
    print(f"Количество эпизодов: {dataset.num_episodes}")
    print(f"Количество кадров: {len(dataset)}")
    return dataset


def create_dataloader(dataset: LeRobotDataset, episode_index: int):
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        sampler=episode_sampler,
    )
    print(f"Выбран эпизод: {episode_index}")
    print(f"Количество кадров в эпизоде: {len(episode_sampler)}")
    return dataloader, episode_sampler


def create_all_episode_dataloaders(dataset: LeRobotDataset):
    episode_loaders = []
    for episode_index in range(dataset.num_episodes):
        episode_loaders.append((episode_index, *create_dataloader(dataset, episode_index)))
    return episode_loaders


def save_stats_json(dataset: LeRobotDataset) -> None:
    stats = serialize_dict(dataset.meta.stats)
    path_stats = dataset.root / "meta" / "stats.json"
    write_json(stats, path_stats)
    print(f"Файл сохранён: {path_stats}")
