from __future__ import annotations

import shutil
from dataclasses import dataclass

import glfw
import numpy as np
from PIL import Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mujoco_env.y_env import SimpleEnv

from .config import CollectDataConfig


@dataclass(slots=True)
class SessionState:
    action: np.ndarray
    episode_id: int = 0
    record_flag: bool = False
    recorded_frames: int = 0


def create_env(config: CollectDataConfig) -> SimpleEnv:
    return SimpleEnv(
        config.xml_path,
        seed=config.seed,
        action_type=config.action_type,
        state_type="joint_angle",
    )


def create_or_load_dataset(config: CollectDataConfig) -> LeRobotDataset:
    create_new = True
    if config.root.exists():
        print(f"Папка {config.root} уже существует.")
        ans = input("Удалить её и создать датасет заново? (y/n) ")
        if ans == "y":
            shutil.rmtree(config.root)
        else:
            create_new = False

    if create_new:
        return LeRobotDataset.create(
            repo_id=config.repo_name,
            root=str(config.root),
            robot_type="so101",
            fps=config.fps,
            features={
                "observation.image": {
                    "dtype": "image",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.wrist_image": {
                    "dtype": "image",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["action"],
                },
                "obj_init": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["obj_init"],
                },
            },
            image_writer_threads=config.image_writer_threads,
            image_writer_processes=config.image_writer_processes,
        )

    print("Загружаю существующий датасет")
    return LeRobotDataset(config.repo_name, root=str(config.root))


def collect_demonstrations(config: CollectDataConfig, env: SimpleEnv, dataset: LeRobotDataset, controller) -> None:
    state = SessionState(action=np.zeros(6, dtype=np.float32))

    while env.env.is_viewer_alive() and state.episode_id < config.num_demo:
        env.step_env()

        if not env.env.loop_every(HZ=config.fps):
            continue

        if config.use_master_arm:
            state.action = controller.get_action()
            reset = env.env.is_key_pressed_once(key=glfw.KEY_Z)
            moved = controller.has_significant_motion(state.action)
        else:
            state.action, reset = env.teleop_robot()
            moved = np.linalg.norm(state.action[:-1]) > 1e-6 or abs(float(state.action[-1])) > 1e-6

        if reset:
            _reset_scene(config, env, dataset, controller, state)
            print("Сцена сброшена, текущий эпизод очищен")
            continue

        if not state.record_flag and moved:
            state.record_flag = True
            state.recorded_frames = 0
            print("Начинаю запись")

        ee_pose = env.get_ee_pose()
        agent_image, wrist_image = env.grab_image()
        agent_image = _resize_image(agent_image, config.image_size)
        wrist_image = _resize_image(wrist_image, config.image_size)

        env.step(state.action)
        commanded_q = np.asarray(env.q, dtype=np.float32).copy()

        if state.record_flag:
            dataset.add_frame(
                {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose,
                    "action": commanded_q,
                    "obj_init": env.obj_init_pose,
                },
                task=config.task_name,
            )
            state.recorded_frames += 1

        if env.env.is_key_pressed_once(key=glfw.KEY_X):
            _handle_manual_save(config, env, dataset, controller, state)
            continue

        done = env.check_success()
        if done and state.record_flag and state.recorded_frames > 0:
            _save_episode(dataset, state, reason="успех")
            if state.episode_id >= config.num_demo:
                break
            _reset_after_save(config, env, controller, state)
            continue

        env.render(teleop=not config.use_master_arm)


def close_env(env: SimpleEnv) -> None:
    if hasattr(env, "env") and env.env.is_viewer_alive():
        env.env.close_viewer()


def _resize_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(image).resize(image_size))


def _reset_scene(config: CollectDataConfig, env: SimpleEnv, dataset: LeRobotDataset, controller, state: SessionState) -> None:
    env.reset(seed=config.seed)
    dataset.clear_episode_buffer()
    state.record_flag = False
    state.recorded_frames = 0
    if controller is not None:
        controller.reset_reference()


def _save_episode(dataset: LeRobotDataset, state: SessionState, reason: str) -> None:
    dataset.save_episode()
    state.episode_id += 1
    print(f"Эпизод {state.episode_id} сохранён ({reason})")


def _reset_after_save(config: CollectDataConfig, env: SimpleEnv, controller, state: SessionState) -> None:
    env.reset(seed=config.seed)
    state.record_flag = False
    state.recorded_frames = 0
    if controller is not None:
        controller.reset_reference()


def _handle_manual_save(
    config: CollectDataConfig,
    env: SimpleEnv,
    dataset: LeRobotDataset,
    controller,
    state: SessionState,
) -> None:
    if state.record_flag and state.recorded_frames > 0:
        _save_episode(dataset, state, reason="по кнопке X")
        if state.episode_id >= config.num_demo:
            return
    else:
        dataset.clear_episode_buffer()
        print("Нажата X, но записывать было нечего — эпизод не сохранён")

    _reset_after_save(config, env, controller, state)
