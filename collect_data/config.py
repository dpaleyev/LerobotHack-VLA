from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CollectDataConfig:
    seed: int | None = 0
    repo_name: str = "so101_pnp"
    num_demo: int = 3
    root: Path = Path("./demo_data")
    use_master_arm: bool = True
    leader_port: str = "/dev/ttyACM0"
    leader_id: str = "None"
    motion_threshold: float = 0.03
    task_name: str = "Put cube on plate"
    xml_path: str = "./asset/example_scene_y.xml"
    fps: int = 20
    image_size: tuple[int, int] = (640, 480)  # (width, height) для PIL.resize
    image_writer_threads: int = 10
    image_writer_processes: int = 5

    @property
    def action_type(self) -> str:
        return "joint_angle" if self.use_master_arm else "delta_joint_angle"


def default_config() -> CollectDataConfig:
    return CollectDataConfig()
