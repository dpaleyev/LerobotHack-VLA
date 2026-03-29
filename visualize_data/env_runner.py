from __future__ import annotations

import numpy as np
from mujoco_env.y_env import SimpleEnv

from .config import VisualizeDataConfig


def create_env(config: VisualizeDataConfig) -> SimpleEnv:
    return SimpleEnv(
        config.xml_path,
        action_type="joint_angle",
        state_type="joint_angle",
    )


def replay_episode(config: VisualizeDataConfig, env: SimpleEnv, dataloader, episode_sampler) -> None:
    step = 0
    iter_dataloader = iter(dataloader)
    env.reset()

    while env.env.is_viewer_alive():
        env.step_env()

        if not env.env.loop_every(HZ=config.fps):
            continue

        try:
            data = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(dataloader)
            env.reset()
            step = 0
            continue

        if step == 0:
            obj_init = data["obj_init"][0].cpu().numpy().astype(np.float32)
            if obj_init.shape[0] >= 14:
                env.set_obj_pose(obj_init[:7], obj_init[7:14])
            else:
                env.set_obj_pose(obj_init[:3], obj_init[3:6])

        action = data["action"][0].cpu().numpy().astype(np.float32)
        _ = env.step(action)

        rgb_agent = _to_uint8_image(data["observation.image"][0].cpu().numpy())
        rgb_wrist = _to_uint8_image(data["observation.wrist_image"][0].cpu().numpy())

        env.rgb_agent = rgb_agent
        env.rgb_ego = rgb_wrist

        env.render()
        step += 1

        if step >= len(episode_sampler):
            iter_dataloader = iter(dataloader)
            env.reset()
            step = 0


def replay_all_episodes(config: VisualizeDataConfig, env: SimpleEnv, episode_loaders) -> None:
    if not episode_loaders:
        raise ValueError("В датасете нет эпизодов для воспроизведения.")

    episode_pos = 0
    step = 0
    _, dataloader, episode_sampler = episode_loaders[episode_pos]
    iter_dataloader = iter(dataloader)
    env.reset()

    while env.env.is_viewer_alive():
        env.step_env()

        if not env.env.loop_every(HZ=config.fps):
            continue

        try:
            data = next(iter_dataloader)
        except StopIteration:
            episode_pos = (episode_pos + 1) % len(episode_loaders)
            episode_index, dataloader, episode_sampler = episode_loaders[episode_pos]
            print(f"Перехожу к эпизоду: {episode_index}")
            iter_dataloader = iter(dataloader)
            env.reset()
            step = 0
            continue

        if step == 0:
            obj_init = data["obj_init"][0].cpu().numpy().astype(np.float32)
            if obj_init.shape[0] >= 14:
                env.set_obj_pose(obj_init[:7], obj_init[7:14])
            else:
                env.set_obj_pose(obj_init[:3], obj_init[3:6])

        action = data["action"][0].cpu().numpy().astype(np.float32)
        _ = env.step(action)

        rgb_agent = _to_uint8_image(data["observation.image"][0].cpu().numpy())
        rgb_wrist = _to_uint8_image(data["observation.wrist_image"][0].cpu().numpy())

        env.rgb_agent = rgb_agent
        env.rgb_ego = rgb_wrist

        env.render()
        step += 1


def close_env(env: SimpleEnv) -> None:
    if hasattr(env, "env") and env.env.is_viewer_alive():
        env.env.close_viewer()


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    return image
