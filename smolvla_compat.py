from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


def _load_raw_config(pretrained_name_or_path: str | Path) -> dict:
    model_path = Path(pretrained_name_or_path)
    if model_path.exists():
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        return json.loads(config_path.read_text())

    config_path = hf_hub_download(str(pretrained_name_or_path), "config.json")
    return json.loads(Path(config_path).read_text())


def load_smolvla_config(
    pretrained_name_or_path: str | Path,
    device: torch.device | None = None,
) -> SmolVLAConfig:
    try:
        cfg = SmolVLAConfig.from_pretrained(str(pretrained_name_or_path))
    except Exception:
        raw = _load_raw_config(pretrained_name_or_path)
        valid_fields = {field.name for field in dataclasses.fields(SmolVLAConfig)}
        cfg_kwargs = {
            key: value
            for key, value in raw.items()
            if key in valid_fields and key not in {"input_features", "output_features", "type"}
        }
        cfg_kwargs["input_features"] = {
            key: PolicyFeature(type=FeatureType[value["type"]], shape=tuple(value["shape"]))
            for key, value in raw.get("input_features", {}).items()
        }
        cfg_kwargs["output_features"] = {
            key: PolicyFeature(type=FeatureType[value["type"]], shape=tuple(value["shape"]))
            for key, value in raw.get("output_features", {}).items()
        }
        cfg_kwargs["normalization_mapping"] = {
            key: NormalizationMode[value] for key, value in raw.get("normalization_mapping", {}).items()
        }
        cfg = SmolVLAConfig(**cfg_kwargs)

    if device is not None:
        cfg.device = device.type
    return cfg
