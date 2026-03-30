from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class VisualizeDataConfig:
    repo_name: str = "so101_pnp"
    root: Path = Path("./demo_data")
    episode: int | Literal["all"] = "all"
    xml_path: str = "./asset/example_scene_y.xml"
    fps: int = 20
    save_stats: bool = False


def default_config() -> VisualizeDataConfig:
    return VisualizeDataConfig()
