import numpy as np
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SO101Leader


class SO101MasterArmController:
    """Мост между реальной мастер-рукой SO-101 и симуляцией MuJoCo.

    Возвращает action в порядке суставов, ожидаемом SimpleEnv(action_type='joint_angle'):
    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

    LeRobot leader для SO-101 отдает значения уже после аппаратной калибровки
    (`None.json`) в нормализованном виде:
    - суставы корпуса: [-100, 100]
    - gripper: [0, 100]

    Здесь эти значения переводятся в диапазоны суставов MuJoCo.
    """

    JOINT_ORDER = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    JOINT_MIN = np.array(
        [
            -1.91986,
            -1.74533,
            -1.69,
            -1.65806,
            -2.74385,
            -0.17453,
        ],
        dtype=np.float32,
    )
    JOINT_MAX = np.array(
        [
            1.91986,
            1.74533,
            1.69,
            1.65806,
            2.84121,
            1.74533,
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        port: str,
        leader_id: str = "None",
        use_degrees: bool = False,
        motion_threshold: float = 0.03,
        sign_map: dict | None = None,
        offset_map: dict | None = None,
    ):

        self.port = port
        self.leader_id = leader_id
        self.use_degrees = use_degrees
        self.motion_threshold = motion_threshold
        self.sign_map = sign_map or {name: 1.0 for name in self.JOINT_ORDER}
        self.offset_map = offset_map or {name: 0.0 for name in self.JOINT_ORDER}
        self.device = None
        self.reference_action = None

    def connect(self):
        cfg = SO101LeaderConfig(
            port=self.port,
            id=self.leader_id,
        )
        self.device = SO101Leader(cfg)
        self.device.connect()
        return self

    def disconnect(self):
        if self.device is not None:
            try:
                self.device.disconnect()
            except Exception:
                pass
            self.device = None

    def reset_reference(self):
        self.reference_action = None

    def _extract_joint_value(self, raw_action: dict, joint_name: str) -> float:
        if joint_name in raw_action:
            return float(raw_action[joint_name])
        key = f"{joint_name}.pos"
        if key in raw_action:
            return float(raw_action[key])
        raise KeyError(
            f"Сустав '{joint_name}' не найден в action leader. Доступные ключи: {list(raw_action.keys())}"
        )

    def _map_leader_value_to_sim(self, joint_name: str, value: float) -> float:
        joint_idx = self.JOINT_ORDER.index(joint_name)
        q_min = float(self.JOINT_MIN[joint_idx])
        q_max = float(self.JOINT_MAX[joint_idx])

        if joint_name == "gripper":
            # LeRobot gripper is normalized to [0, 100].
            normalized = float(np.clip(value, 0.0, 100.0))
            sim_value = q_min + (normalized / 100.0) * (q_max - q_min)
        else:
            # LeRobot body joints are normalized to [-100, 100].
            normalized = float(np.clip(value, -100.0, 100.0))
            sim_value = q_min + ((normalized + 100.0) / 200.0) * (q_max - q_min)

        return sim_value

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action.astype(np.float32), self.JOINT_MIN, self.JOINT_MAX)

    def get_action(self) -> np.ndarray:
        if self.device is None:
            raise RuntimeError("Мастер-рука не подключена. Сначала вызови connect().")

        raw_action = self.device.get_action()
        out = []
        for joint_name in self.JOINT_ORDER:
            leader_value = self._extract_joint_value(raw_action, joint_name)
            sim_value = self._map_leader_value_to_sim(joint_name, leader_value)
            sim_value = (
                self.sign_map[joint_name] * sim_value + self.offset_map[joint_name]
            )
            out.append(sim_value)

        action = self._clip_action(np.array(out, dtype=np.float32))
        if self.reference_action is None:
            self.reference_action = action.copy()
        return action

    def has_significant_motion(self, action: np.ndarray) -> bool:
        if self.reference_action is None:
            self.reference_action = action.copy()
            return False
        return (
            float(np.linalg.norm(action - self.reference_action))
            > self.motion_threshold
        )
