from __future__ import annotations

from pathlib import Path

from collect_data.config import default_config

DEFAULT_PRETRAINED_POLICY = "lerobot/smolvla_base"
DEFAULT_DATASET_REPO_ID = "so101_pnp"
DEFAULT_OUTPUT_DIR = Path("./outputs/train/so101_smolvla_safe")
DEFAULT_OFFICIAL_TRAIN_DIR = Path("./outputs/train/so101_smolvla_official_main_bs32_lr1e4_noamp")


def default_dataset_root() -> Path:
    candidates = (
        Path("./demo_data_merged_draft_hf"),
        Path("./demo_data_merged_badness_le2_hf"),
        default_config().root,
        Path("./demo_data"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR


def _checkpoint_pretrained_dirs(train_run_dir: Path) -> list[Path]:
    checkpoints_dir = train_run_dir / "checkpoints"
    candidates = [checkpoints_dir / "last" / "pretrained_model"]

    if checkpoints_dir.exists():
        numeric_dirs = sorted(
            (path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
            reverse=True,
        )
        candidates.extend(path / "pretrained_model" for path in numeric_dirs)

    return candidates


def default_train_run_dir() -> Path:
    return DEFAULT_OFFICIAL_TRAIN_DIR


def default_train_config_path() -> Path:
    candidates = [
        pretrained_dir / "train_config.json"
        for train_run_dir in (default_train_run_dir(), default_output_dir())
        for pretrained_dir in _checkpoint_pretrained_dirs(train_run_dir)
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return default_train_run_dir() / "checkpoints" / "last" / "pretrained_model" / "train_config.json"


def default_model_path() -> Path:
    candidates = [
        pretrained_dir
        for train_run_dir in (default_train_run_dir(), default_output_dir())
        for pretrained_dir in _checkpoint_pretrained_dirs(train_run_dir)
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return default_train_run_dir() / "checkpoints" / "last" / "pretrained_model"
