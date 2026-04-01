from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CollectDataConfig:
    seed: int | None = 0
    repo_name: str = "so101_pnp"
    num_demo: int = 150
    root: Path = Path("./simulation_data_fixed")
    use_master_arm: bool = True
    leader_port: str = "/dev/ttyACM0"
    leader_id: str = "my_leader"
    motion_threshold: float = 0.03
    task_name: str = "Put cube on plate"
    xml_path: str = "./asset/example_scene_y.xml"
    fps: int = 10
    image_size: tuple[int, int] = (640, 480)  # (width, height) для PIL.resize
    image_writer_threads: int = 10
    image_writer_processes: int = 5
    batch_encoding_size: int = 1
    vcodec: str = "h264"
    metadata_buffer_size: int = 10
    streaming_encoding: bool = True
    encoder_threads: int | None = None
    state_contract: str = "joint_pos"

    def __post_init__(self) -> None:
        if self.state_contract != "joint_pos":
            raise ValueError(
                "CollectDataConfig.state_contract поддерживает только canonical значение "
                "'joint_pos' (совместимо с lerobot-record / so_follower)."
            )

    @property
    def action_type(self) -> str:
        return "joint_angle" if self.use_master_arm else "delta_joint_angle"


def default_config() -> CollectDataConfig:
    return CollectDataConfig()
