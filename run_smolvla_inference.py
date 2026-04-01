#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from collect_data.config import default_config
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType
from smolvla_compat import load_smolvla_config
from smolvla_defaults import (
    DEFAULT_DATASET_REPO_ID,
    default_dataset_root,
    default_train_config_path,
    default_train_run_dir,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent
FRONT_CAMERA_KEYS = {"observation.image", "observation.images.front", "observation.images.camera1"}
SIDE_CAMERA_KEYS = {"observation.wrist_image", "observation.images.side", "observation.images.camera2"}
ZERO_CAMERA_KEYS = {"observation.images.camera3"}
EMPTY_CAMERA_PREFIX = "observation.images.empty_camera_"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
EXPECTED_STATE_SHAPE = (6,)
EXPECTED_ACTION_SHAPE = (6,)


@dataclasses.dataclass(frozen=True)
class VisualFeatureSpec:
    key: str
    source: str
    shape: tuple[int, int, int]


@dataclasses.dataclass(frozen=True)
class InferenceArtifacts:
    policy_path: Path
    train_config_path: Path | None
    train_config: dict[str, Any] | None
    dataset_root: Path
    dataset_repo_id: str


@dataclasses.dataclass(frozen=True)
class NumericFeatureContract:
    state_unit: str
    action_unit: str


JOINT_POSITION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
GRIPPER_MIN_RAD = -0.17453
GRIPPER_MAX_RAD = 1.74533
GRIPPER_MIN_REAL = 0.0
GRIPPER_MAX_REAL = 100.0


def parse_args():
    cfg = default_config()
    parser = argparse.ArgumentParser(description="SmolVLA inference in MuJoCo.")
    parser.add_argument("--policy-path", "--model", dest="policy_path", type=Path, default=None)
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--train-run-dir", type=Path, default=default_train_run_dir())
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--task", default=cfg.task_name)
    parser.add_argument("--xml-path", default=cfg.xml_path)
    parser.add_argument("--fps", type=int, default=cfg.fps)
    parser.add_argument("--seed", type=int, default=cfg.seed or 0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--action-alpha",
        type=float,
        default=1.0,
        help="EMA coefficient for action smoothing. 1.0 disables smoothing.",
    )
    parser.add_argument(
        "--max-joint-step-deg",
        type=float,
        default=None,
        help="Maximum per-step change for arm joints in degrees.",
    )
    parser.add_argument(
        "--max-gripper-step",
        type=float,
        default=None,
        help="Maximum per-step change for gripper command in radians.",
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def get_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def remap_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None

    path = Path(path)
    if not path.is_absolute():
        return WORKSPACE_ROOT / path
    if path == Path("/app"):
        return WORKSPACE_ROOT

    parts = path.parts
    if len(parts) >= 2 and parts[1] == "app":
        return WORKSPACE_ROOT.joinpath(*parts[2:])
    return path


def checkpoint_dir(train_run_dir: Path, checkpoint_step: int) -> Path:
    return train_run_dir / "checkpoints" / f"{checkpoint_step:06d}" / "pretrained_model"


def load_train_config(train_config_path: Path | None) -> dict[str, Any] | None:
    if train_config_path is None:
        return None
    return json.loads(train_config_path.read_text())


def resolve_artifacts(args) -> InferenceArtifacts:
    train_config_path: Path | None
    policy_path: Path

    if args.checkpoint_step is not None:
        train_run_dir = remap_path(args.train_run_dir)
        if train_run_dir is None:
            raise ValueError("Train run directory is required when --checkpoint-step is used.")
        policy_path = checkpoint_dir(train_run_dir, args.checkpoint_step)
        train_config_path = policy_path / "train_config.json"
        if not train_config_path.exists():
            raise FileNotFoundError(f"Train config not found for checkpoint: {train_config_path}")
    elif args.train_config is not None:
        train_config_path = remap_path(args.train_config)
        if train_config_path is None or not train_config_path.exists():
            raise FileNotFoundError(f"Train config not found: {train_config_path}")
        policy_path = train_config_path.parent
    elif args.policy_path is not None:
        policy_path = remap_path(args.policy_path)
        if policy_path is None:
            raise ValueError("Policy path is required.")
        candidate = policy_path / "train_config.json"
        train_config_path = candidate if candidate.exists() else None
    else:
        train_config_path = remap_path(default_train_config_path())
        if train_config_path is None or not train_config_path.exists():
            raise FileNotFoundError(
                "Default official train_config.json not found. "
                "Pass --train-config, --policy-path, or --checkpoint-step."
            )
        policy_path = train_config_path.parent

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")

    train_config = load_train_config(train_config_path)
    dataset_cfg = (train_config or {}).get("dataset", {})
    dataset_root = remap_path(args.dataset_root or dataset_cfg.get("root")) or default_dataset_root()
    dataset_repo_id = args.dataset_repo_id or dataset_cfg.get("repo_id") or DEFAULT_DATASET_REPO_ID

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. "
            "Pass --dataset-root explicitly if the train config points to a container path."
        )

    return InferenceArtifacts(
        policy_path=policy_path,
        train_config_path=train_config_path,
        train_config=train_config,
        dataset_root=dataset_root,
        dataset_repo_id=dataset_repo_id,
    )


def load_policy(model_path: Path, dataset_root: Path, dataset_repo_id: str, device: torch.device):
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=str(dataset_root))
    cfg = load_smolvla_config(model_path, device=device)
    if model_path.is_dir() and cfg.load_vlm_weights:
        cfg.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(
        str(model_path),
        config=cfg,
        dataset_stats=ds_meta.stats,
        strict=False,
    )
    if device.type != "cuda":
        policy.float()
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(model_path),
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": ds_meta.stats,
                "features": {**cfg.input_features, **cfg.output_features},
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": ds_meta.stats,
                "features": {**cfg.input_features, **cfg.output_features},
            },
        },
    )
    return policy, cfg, preprocessor, postprocessor


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def infer_numeric_contract(dataset_root: Path) -> NumericFeatureContract:
    info = load_json_if_exists(dataset_root / "meta" / "info.json") or {}
    stats = load_json_if_exists(dataset_root / "meta" / "stats.json") or {}
    features = info.get("features", {})

    state_names = tuple(features.get(STATE_KEY, {}).get("names") or ())
    action_names = tuple(features.get(ACTION_KEY, {}).get("names") or ())

    if state_names != JOINT_POSITION_NAMES or action_names != JOINT_POSITION_NAMES:
        raise ValueError(
            "Dataset numeric contract is not canonical so_follower joint .pos. "
            "Expected observation.state/action names to match JOINT_POSITION_NAMES."
        )

    def infer_unit(feature_key: str) -> str:
        feature_stats = stats.get(feature_key, {})
        mins = np.asarray(feature_stats.get("min", []), dtype=np.float32).reshape(-1)
        maxs = np.asarray(feature_stats.get("max", []), dtype=np.float32).reshape(-1)
        if mins.size == 0 and maxs.size == 0:
            return "radians"
        abs_max = float(np.max(np.concatenate([np.abs(mins), np.abs(maxs)])))
        return "degrees" if abs_max > 10.0 else "radians"

    state_unit = infer_unit(STATE_KEY)
    action_unit = infer_unit(ACTION_KEY)

    if action_unit == "degrees":
        state_unit = "degrees"

    return NumericFeatureContract(
        state_unit=state_unit,
        action_unit=action_unit,
    )


def visual_source_for_key(key: str) -> str | None:
    if key in FRONT_CAMERA_KEYS:
        return "front"
    if key in SIDE_CAMERA_KEYS:
        return "side"
    if key in ZERO_CAMERA_KEYS or key.startswith(EMPTY_CAMERA_PREFIX):
        return "zeros"
    return None


def validate_feature_contract(model_cfg) -> list[VisualFeatureSpec]:
    visual_specs: list[VisualFeatureSpec] = []

    for key, feature in model_cfg.input_features.items():
        shape = tuple(int(dim) for dim in feature.shape)

        if feature.type == FeatureType.VISUAL:
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"Unsupported visual feature shape for {key}: {shape}")
            source = visual_source_for_key(key)
            if source is None:
                raise ValueError(
                    f"Unsupported visual feature key {key}. "
                    "Supported keys are front/wrist aliases plus empty-camera placeholders."
                )
            visual_specs.append(VisualFeatureSpec(key=key, source=source, shape=shape))
            continue

        if feature.type == FeatureType.STATE:
            if key != STATE_KEY:
                raise ValueError(f"Unsupported state feature key: {key}")
            if shape != EXPECTED_STATE_SHAPE:
                raise ValueError(
                    f"Unsupported state feature shape for {key}: {shape}. "
                    f"Expected {EXPECTED_STATE_SHAPE} for canonical 6D state."
                )
            continue

        raise ValueError(f"Unsupported input feature type for {key}: {feature.type}")

    if STATE_KEY not in model_cfg.input_features:
        raise ValueError(f"Policy is missing required state feature: {STATE_KEY}")
    if not visual_specs:
        raise ValueError("Policy does not define any visual input features.")

    action_feature = model_cfg.output_features.get(ACTION_KEY)
    if action_feature is None:
        raise ValueError(f"Policy is missing required output feature: {ACTION_KEY}")
    action_shape = tuple(int(dim) for dim in action_feature.shape)
    if action_feature.type != FeatureType.ACTION:
        raise ValueError(f"Unsupported output feature type for {ACTION_KEY}: {action_feature.type}")
    if action_shape != EXPECTED_ACTION_SHAPE:
        raise ValueError(
            f"Unsupported action shape for {ACTION_KEY}: {action_shape}. "
            f"Expected {EXPECTED_ACTION_SHAPE} for joint-angle control."
        )

    return visual_specs


def _grab_images_with_offscreen_renderer(env) -> tuple[np.ndarray, np.ndarray]:
    """Headless-safe camera capture when MuJoCo viewer is unavailable."""
    renderers = getattr(env, "_offscreen_renderers", None)
    if renderers is None:
        # Match the dataset camera aspect; exact size is resized later per feature spec.
        renderers = {
            "front": mujoco.Renderer(env.env.model, height=480, width=640),
            "side": mujoco.Renderer(env.env.model, height=480, width=640),
        }
        setattr(env, "_offscreen_renderers", renderers)

    front_renderer = renderers["front"]
    side_renderer = renderers["side"]
    front_renderer.update_scene(env.env.data, camera=env.agent_cam_name)
    side_renderer.update_scene(env.env.data, camera=env.ego_cam_name)
    front_rgb = front_renderer.render().copy()
    side_rgb = side_renderer.render().copy()
    return front_rgb, side_rgb


def _map_gripper_rad_to_real(gripper_rad: float) -> float:
    gripper_rad = float(np.clip(gripper_rad, GRIPPER_MIN_RAD, GRIPPER_MAX_RAD))
    alpha = (gripper_rad - GRIPPER_MIN_RAD) / (GRIPPER_MAX_RAD - GRIPPER_MIN_RAD)
    return float(alpha * (GRIPPER_MAX_REAL - GRIPPER_MIN_REAL) + GRIPPER_MIN_REAL)


def _map_gripper_real_to_rad(gripper_real: float) -> float:
    gripper_real = float(np.clip(gripper_real, GRIPPER_MIN_REAL, GRIPPER_MAX_REAL))
    alpha = (gripper_real - GRIPPER_MIN_REAL) / (GRIPPER_MAX_REAL - GRIPPER_MIN_REAL)
    return float(alpha * (GRIPPER_MAX_RAD - GRIPPER_MIN_RAD) + GRIPPER_MIN_RAD)


def make_observation(
    env,
    task: str,
    visual_specs: list[VisualFeatureSpec],
    device: torch.device,
    numeric_contract: NumericFeatureContract,
) -> dict[str, torch.Tensor | list[str]]:
    try:
        front, side = env.grab_image()
    except AttributeError as exc:
        # In headless mode SimpleEnv skips viewer init; use offscreen renderer instead.
        if "viewer" not in str(exc):
            raise
        front, side = _grab_images_with_offscreen_renderer(env)
    to_tensor = ToTensor()
    resized_cache: dict[tuple[str, int, int], torch.Tensor] = {}
    obs: dict[str, torch.Tensor | list[str]] = {"task": [task]}

    for spec in visual_specs:
        _, height, width = spec.shape
        cache_key = (spec.source, width, height)

        if spec.source == "zeros":
            tensor = torch.zeros(spec.shape, dtype=torch.float32)
        else:
            image = front if spec.source == "front" else side
            if cache_key not in resized_cache:
                resized_cache[cache_key] = to_tensor(Image.fromarray(image).resize((width, height)))
            tensor = resized_cache[cache_key]

        obs[spec.key] = tensor.unsqueeze(0).to(device)

    state = env.get_joint_state()

    if numeric_contract.state_unit == "degrees":
        state = np.asarray(state, dtype=np.float32)
        state[:5] = np.rad2deg(state[:5]).astype(np.float32)
        state[5] = _map_gripper_rad_to_real(float(state[5]))

    obs[STATE_KEY] = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    return obs


def stabilize_action(
    action: np.ndarray,
    previous_action: np.ndarray,
    action_alpha: float,
    max_joint_step: float | None,
    max_gripper_step: float | None,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    previous_action = np.asarray(previous_action, dtype=np.float32)

    if not (0.0 < action_alpha <= 1.0):
        raise ValueError(f"--action-alpha must be in (0, 1], got {action_alpha}")

    if action_alpha < 1.0:
        action = action_alpha * action + (1.0 - action_alpha) * previous_action

    delta = action - previous_action
    if max_joint_step is not None:
        delta[:-1] = np.clip(delta[:-1], -max_joint_step, max_joint_step)
    if max_gripper_step is not None:
        delta[-1] = np.clip(delta[-1], -max_gripper_step, max_gripper_step)

    return (previous_action + delta).astype(np.float32)


def run_episode(
    env,
    policy: SmolVLAPolicy,
    preprocessor,
    postprocessor,
    task: str,
    visual_specs: list[VisualFeatureSpec],
    fps: int,
    max_steps: int,
    device: torch.device,
    render: bool,
    numeric_contract: NumericFeatureContract,
    action_alpha: float,
    max_joint_step: float | None,
    max_gripper_step: float | None,
) -> dict[str, int | bool]:
    policy.reset()
    control_step = 0
    previous_action = np.asarray(env.q, dtype=np.float32).copy()

    while control_step < max_steps:
        env.step_env()
        if not env.env.loop_every(HZ=fps):
            continue

        obs = make_observation(env, task, visual_specs, device, numeric_contract)
        obs = preprocessor(obs)
        with torch.no_grad():
            action = policy.select_action(obs)
        action = postprocessor(action)
        action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if numeric_contract.action_unit == "degrees":
            action_np = np.asarray(action_np, dtype=np.float32)
            action_np[:5] = np.deg2rad(action_np[:5]).astype(np.float32)
            action_np[5] = _map_gripper_real_to_rad(float(action_np[5]))
        action_np = stabilize_action(
            action=action_np,
            previous_action=previous_action,
            action_alpha=action_alpha,
            max_joint_step=max_joint_step,
            max_gripper_step=max_gripper_step,
        )
        previous_action = action_np.copy()
        env.step(action_np)

        if render:
            try:
                env.render()
            except Exception:
                pass

        control_step += 1
        if env.check_success():
            return {"success": True, "steps": control_step}

    return {"success": False, "steps": control_step}


def main():
    args = parse_args()
    device = get_device(args.device)
    collect_cfg = default_config()
    artifacts = resolve_artifacts(args)
    max_joint_step = (
        None if args.max_joint_step_deg is None else np.deg2rad(args.max_joint_step_deg)
    )

    if args.headless:
        os.environ["MUJOCO_GLFW_VISIBLE"] = "0"

    from mujoco_env.y_env import SimpleEnv

    print(f"Device: {device}")
    print(f"Policy: {artifacts.policy_path}")
    print(f"Train config: {artifacts.train_config_path or 'not provided'}")
    print(f"Dataset root: {artifacts.dataset_root}")
    print(f"Dataset repo id: {artifacts.dataset_repo_id}")
    print(f"Task: {args.task}")
    print(f"Headless: {args.headless}")
    print(
        "Action stabilization:",
        f"alpha={args.action_alpha},",
        f"max_joint_step_deg={args.max_joint_step_deg},",
        f"max_gripper_step={args.max_gripper_step}",
    )

    policy, model_cfg, preprocessor, postprocessor = load_policy(
        artifacts.policy_path,
        artifacts.dataset_root,
        artifacts.dataset_repo_id,
        device,
    )
    visual_specs = validate_feature_contract(model_cfg)
    numeric_contract = infer_numeric_contract(artifacts.dataset_root)
    print(
        "Policy config:",
        f"chunk_size={model_cfg.chunk_size},",
        f"n_action_steps={model_cfg.n_action_steps},",
        f"empty_cameras={model_cfg.empty_cameras},",
        f"visual_keys={[spec.key for spec in visual_specs]}",
    )
    print(
        "Numeric contract:",
        "state_source=joint_state,",
        f"state_unit={numeric_contract.state_unit},",
        f"action_unit={numeric_contract.action_unit}",
    )

    env = SimpleEnv(
        args.xml_path,
        action_type=collect_cfg.action_type,
        state_type="joint_angle",
        seed=args.seed,
        visible_window=not args.headless,
    )

    results: list[dict[str, int | bool]] = []
    for episode_idx in range(args.episodes):
        env.reset(seed=args.seed + episode_idx)
        result = run_episode(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=args.task,
            visual_specs=visual_specs,
            fps=args.fps,
            max_steps=args.max_steps,
            device=device,
            render=not args.headless,
            numeric_contract=numeric_contract,
            action_alpha=args.action_alpha,
            max_joint_step=max_joint_step,
            max_gripper_step=args.max_gripper_step,
        )
        results.append(result)
        status = "SUCCESS" if result["success"] else "TIMEOUT"
        print(f"Episode {episode_idx}: {status} in {result['steps']} steps")

    success_count = sum(int(item["success"]) for item in results)
    summary = {
        "policy_path": str(artifacts.policy_path),
        "train_config_path": str(artifacts.train_config_path) if artifacts.train_config_path else None,
        "dataset_root": str(artifacts.dataset_root),
        "dataset_repo_id": artifacts.dataset_repo_id,
        "numeric_contract": dataclasses.asdict(numeric_contract),
        "task": args.task,
        "episodes": args.episodes,
        "successes": success_count,
        "success_rate": success_count / max(args.episodes, 1),
        "avg_steps": float(np.mean([item["steps"] for item in results])) if results else 0.0,
        "results": results,
    }

    print(json.dumps(summary, indent=2))

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    if hasattr(env.env, "viewer") and env.env.is_viewer_alive():
        env.env.close_viewer()


if __name__ == "__main__":
    main()
