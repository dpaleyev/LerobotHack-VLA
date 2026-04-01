from __future__ import annotations

import json
import shutil
from dataclasses import dataclass

import glfw
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mujoco_env.y_env import SimpleEnv

from .config import CollectDataConfig

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
REQUIRED_FEATURE_KEYS = {
    "observation.images.front",
    "observation.images.side",
    "observation.state",
    "action",
}
GRIPPER_MIN_RAD = -0.17453
GRIPPER_MAX_RAD = 1.74533
GRIPPER_MIN_REAL = 0.0
GRIPPER_MAX_REAL = 100.0


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
            robot_type="so_follower",
            fps=config.fps,
            features={
                "observation.images.front": {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.images.side": {
                    "dtype": "video",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": JOINT_NAMES,
                },
                "action": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": JOINT_NAMES,
                },
            },
            use_videos=True,
            image_writer_threads=config.image_writer_threads,
            image_writer_processes=config.image_writer_processes,
            batch_encoding_size=config.batch_encoding_size,
            vcodec=config.vcodec,
            metadata_buffer_size=config.metadata_buffer_size,
            streaming_encoding=config.streaming_encoding,
            encoder_threads=config.encoder_threads,
        )

    print("Загружаю существующий датасет")
    info_path = config.root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Не найден metadata-файл датасета: {info_path}")

    info = json.loads(info_path.read_text())
    existing_keys = set(info.get("features", {}))
    state_feature = info.get("features", {}).get("observation.state", {})
    action_feature = info.get("features", {}).get("action", {})
    state_names = list(state_feature.get("names") or [])
    action_names = list(action_feature.get("names") or [])
    state_shape = tuple(int(x) for x in (state_feature.get("shape") or ()))
    action_shape = tuple(int(x) for x in (action_feature.get("shape") or ()))

    if info.get("robot_type") != "so_follower" or not REQUIRED_FEATURE_KEYS.issubset(existing_keys):
        raise ValueError(
            "Существующий датасет записан в старом формате и несовместим с новой схемой записи. "
            "Удалите папку и создайте датасет заново, либо продолжайте писать в новый root."
        )
    if state_shape != (6,) or action_shape != (6,) or state_names != JOINT_NAMES or action_names != JOINT_NAMES:
        raise ValueError(
            "Существующий датасет не соответствует canonical lerobot-record контракту "
            "для SO follower (state/action должны быть joint .pos, shape=(6,)). "
            "Используйте новый root для записи."
        )

    return LeRobotDataset.resume(
        config.repo_name,
        root=str(config.root),
        image_writer_threads=config.image_writer_threads,
        image_writer_processes=config.image_writer_processes,
        batch_encoding_size=config.batch_encoding_size,
        vcodec=config.vcodec,
        streaming_encoding=config.streaming_encoding,
        encoder_threads=config.encoder_threads,
    )


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

        joint_state = _state_in_real_units(env)
        agent_image, wrist_image = env.grab_image()
        agent_image = _resize_image(agent_image, config.image_size)
        wrist_image = _resize_image(wrist_image, config.image_size)

        env.step(state.action)
        commanded_q = _action_in_real_units(np.asarray(env.q, dtype=np.float32).copy())

        if state.record_flag:
            dataset.add_frame(
                {
                    "observation.images.front": agent_image,
                    "observation.images.side": wrist_image,
                    "observation.state": joint_state,
                    "action": commanded_q,
                    "task": config.task_name,
                }
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


def _map_gripper_rad_to_real(gripper_rad: float) -> float:
    gripper_rad = float(np.clip(gripper_rad, GRIPPER_MIN_RAD, GRIPPER_MAX_RAD))
    alpha = (gripper_rad - GRIPPER_MIN_RAD) / (GRIPPER_MAX_RAD - GRIPPER_MIN_RAD)
    return float(alpha * (GRIPPER_MAX_REAL - GRIPPER_MIN_REAL) + GRIPPER_MIN_REAL)


def _state_in_real_units(env: SimpleEnv) -> np.ndarray:
    # Canonical so_follower units:
    # joints in degrees + gripper in 0..100 range.
    arm_rad = np.asarray(env.env.get_qpos_joints(joint_names=env.arm_joint_names), dtype=np.float32)
    arm_deg = np.rad2deg(arm_rad).astype(np.float32)
    gripper_rad = float(env._get_gripper_q())
    gripper_real = _map_gripper_rad_to_real(gripper_rad)
    return np.concatenate([arm_deg, np.array([gripper_real], dtype=np.float32)], dtype=np.float32)


def _action_in_real_units(action_rad: np.ndarray) -> np.ndarray:
    action_rad = np.asarray(action_rad, dtype=np.float32)
    arm_deg = np.rad2deg(action_rad[:-1]).astype(np.float32)
    gripper_real = _map_gripper_rad_to_real(float(action_rad[-1]))
    return np.concatenate([arm_deg, np.array([gripper_real], dtype=np.float32)], dtype=np.float32)


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
