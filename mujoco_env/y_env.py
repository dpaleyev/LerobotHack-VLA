import os
import random
import copy
import colorsys
import numpy as np
import glfw
import mujoco

from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import quat2r, r2quat, rpy2r, r2rpy


class SimpleEnv:
    def __init__(
        self,
        xml_path,
        action_type="eef_pose",
        state_type="joint_angle",
        seed=None,
        visible_window=True,
    ):
        """
        Args:
            xml_path: str, path to the xml file
            action_type: str, one of:
                - 'eef_pose'         : action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
                - 'delta_joint_angle': action = [dq1, dq2, dq3, dq4, dq5, gripper]
                - 'joint_angle'      : action = [q1, q2, q3, q4, q5, gripper]
            state_type: str, one of:
                - 'joint_angle'
                - 'ee_pose'
                - 'delta_q'
            seed: int or None
            visible_window: показывать ли GLFW-окно MuJoCo. В Jupyter/Cursor при падении ядра
                попробуйте False (рендер камер для датасета сохраняется).
                Либо до импорта mujoco_env: os.environ[\"MUJOCO_GLFW_VISIBLE\"] = \"0\".
        """
        if os.environ.get("MUJOCO_GLFW_VISIBLE", "").strip() == "0":
            visible_window = False
        self._visible_window = visible_window
        self.env = MuJoCoParserClass(name="Tabletop", rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type

        # SO-101 arm joints used for IK/control
        self.arm_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
        self.gripper_joint_name = "gripper"
        self.joint_names = self.arm_joint_names + [self.gripper_joint_name]

        # End-effector body/site in SO-101
        self.ee_body_name = "gripper"

        # Две камеры на столе (object_table.xml): cam_front / cam_side
        self.agent_cam_name = "cam_front"
        self.ego_cam_name = "cam_side"

        # Full control vector [5 arm joints + 1 gripper]
        self.q = np.zeros(len(self.joint_names), dtype=np.float32)
        self.last_q = np.zeros(len(self.arm_joint_names), dtype=np.float32)
        self.compute_q = np.zeros(len(self.arm_joint_names), dtype=np.float32)

        self.gripper_state = False
        self.past_chars = []
        self._episode_counter = 0
        self.cube_half_extent = 0.015

        self.init_viewer()
        self.reset(seed)

    def _as_scalar(self, x):
        x = np.asarray(x).reshape(-1)
        return float(x[0])

    def _get_gripper_q(self):
        return self._as_scalar(self.env.get_qpos_joint(self.gripper_joint_name))

    def _gripper_cmd_value(self):
        return 1.0 if self._get_gripper_q() > 0.2 else 0.0

    def init_viewer(self):
        """Initialize the viewer."""
        self.env.reset()
        if not self._visible_window:
            self.env.use_mujoco_viewer = False
            return
        self.env.init_viewer(
            distance=2.0,
            elevation=-30,
            transparent=False,
            black_sky=True,
            use_rgb_overlay=False,
            loc_rgb_overlay="top right",
            visible_window=self._visible_window,
        )

    def _apply_cube_half_extent(self, half: float):
        """Размер куба (половина ребра) и сайты success; масса ∝ объёму."""
        m, d = self.env.model, self.env.data
        half = float(np.clip(half, 0.01, 0.025))
        self.cube_half_extent = half
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "small_cube_geom")
        if gid < 0:
            return
        ref_half = 0.015
        m.geom_size[gid] = [half, half, half]
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "body_obj_cube")
        if bid >= 0:
            m.body_mass[bid] = 0.1 * (half / ref_half) ** 3
        for sname, z in (("bottom_site_cube", -half), ("top_site_cube", half)):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid >= 0:
                m.site_pos[sid] = [0.0, 0.0, z]
        mujoco.mj_setConst(m, d)
        mujoco.mj_forward(m, d)

    def _randomize_cube_color(self, rng):
        """Случайный насыщенный цвет куба (материал cube_wood в cube.xml)."""
        m = self.env.model
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "cube_wood")
        if mid < 0:
            return
        h = float(rng.uniform(0.0, 1.0))
        s = float(rng.uniform(0.55, 1.0))
        v = float(rng.uniform(0.45, 1.0))
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        m.mat_rgba[mid, 0] = r
        m.mat_rgba[mid, 1] = g
        m.mat_rgba[mid, 2] = b
        m.mat_rgba[mid, 3] = 1.0

    def reset(self, seed=None):
        """
        Reset the environment:
        - set robot to a comfortable initial pose
        - randomize cube size (half-edge 0.01…0.25) and plate/cube poses in reach
        """
        self._episode_counter += 1
        if seed is not None:
            np.random.seed(seed)
        # Отдельный RNG на эпизод: при фиксированном seed каждый сброс даёт новую выкладку
        if seed is not None:
            rng = np.random.default_rng(int(seed) + 1_000_003 * self._episode_counter)
        else:
            rng = np.random.default_rng()

        # Стартовая поза SO-101:
        # 1) shoulder_pan = 0 -> нижний сервопривод строго по центру
        # 2) arm слегка приподнята
        # 3) wrist_roll = 0 -> клешня смотрит в ту же сторону, что и base
        q_zero = np.deg2rad([
            0.0,    # shoulder_pan
            -30.0,  # shoulder_lift
            75.0,   # elbow_flex
            -45.0,  # wrist_flex
            0.0,    # wrist_roll
        ]).astype(np.float32)

        gripper_init = np.array([0.0], dtype=np.float32)
        q_full = np.concatenate([q_zero, gripper_init])

        # Сразу ставим робота в эту позу, без IK
        self.env.forward(q=q_full, joint_names=self.joint_names, increase_tick=False)

        # Случайный размер куба (половина ребра), затем позы объектов в зоне досягаемости
        cube_half = float(rng.uniform(0.01, 0.017))
        self._apply_cube_half_extent(cube_half)
        self._randomize_cube_color(rng)

        obj_names = self.env.get_body_names(prefix="body_obj_")
        n_obj = len(obj_names)
        # SO-101: база робота x≈0.06; min safe 2D distance from base = 0.20 m.
        # x∈[0.22, 0.38], y∈[-0.18, 0.18] → все объекты гарантированно
        # дальше ~16 см от базы по X и >20 см по 2D-норме.
        ROBOT_BASE_XY = np.array([0.06, 0.0])
        MIN_DIST_FROM_BASE = 0.20
        max_attempts = 200
        for attempt in range(max_attempts):
            obj_xyzs = sample_xyzs(
                n_obj,
                x_range=[0.22, 0.38],
                y_range=[-0.18, 0.18],
                z_range=[0.82, 0.82],
                min_dist=0.14,
                xy_margin=0.02,
                rng=rng,
            )
            dists = np.linalg.norm(obj_xyzs[:, :2] - ROBOT_BASE_XY, axis=1)
            if np.all(dists >= MIN_DIST_FROM_BASE):
                break
        else:
            # fallback — просто используем последний сэмпл
            pass

        for obj_idx in range(n_obj):
            self.env.set_p_base_body(body_name=obj_names[obj_idx], p=obj_xyzs[obj_idx, :])
            yaw = float(rng.uniform(0.0, 2.0 * np.pi))
            self.env.set_R_base_body(
                body_name=obj_names[obj_idx],
                R=rpy2r(np.array([0.0, 0.0, yaw], dtype=np.float64)),
            )

        self.env.forward(increase_tick=False)

        self.last_q = copy.deepcopy(q_zero)
        self.compute_q = copy.deepcopy(q_zero)
        self.q = q_full.astype(np.float32)

        self.p0, self.R0 = self.env.get_pR_body(body_name=self.ee_body_name)

        cube_init_pose, plate_init_pose = self.get_obj_pose()
        self.obj_init_pose = np.concatenate(
            [cube_init_pose, plate_init_pose], dtype=np.float32
        )

        for _ in range(100):
            self.step_env()

        self.gripper_state = False
        self.past_chars = []
        print("DONE INITIALIZATION")

    def step(self, action):
        """
        Take a step in the environment.

        For SO-101:
        - eef_pose: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        - delta_joint_angle: [dq1, dq2, dq3, dq4, dq5, gripper]
        - joint_angle: [q1, q2, q3, q4, q5, gripper]
        """
        action = np.asarray(action, dtype=np.float32)

        # Joint limits for SO-101 arm
        q_min = np.array([
            -1.91986,   # shoulder_pan
            -1.74533,   # shoulder_lift
            -1.69,      # elbow_flex
            -1.65806,   # wrist_flex
            -2.74385,   # wrist_roll
        ], dtype=np.float32)

        q_max = np.array([
             1.91986,
             1.74533,
             1.69,
             1.65806,
             2.84121,
        ], dtype=np.float32)

        gripper_min = -0.17453
        gripper_max =  1.74533

        if self.action_type == "eef_pose":
            if action.shape[0] != 7:
                raise ValueError(f"eef_pose action must have shape (7,), got {action.shape}")

            q = self.env.get_qpos_joints(joint_names=self.arm_joint_names)
            self.p0 = self.p0 + action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))

            q, ik_err_stack, ik_info = solve_ik(
                env=self.env,
                joint_names_for_ik=self.arm_joint_names,
                body_name_trgt=self.ee_body_name,
                q_init=q,
                p_trgt=self.p0,
                R_trgt=self.R0,
                max_ik_tick=50,
                ik_stepsize=1.0,
                ik_eps=1e-2,
                ik_th=np.radians(5.0),
                render=False,
                verbose_warning=False,
            )
            q = np.clip(q, q_min, q_max)
            gripper_cmd = np.array([np.clip(action[-1], gripper_min, gripper_max)], dtype=np.float32)

        elif self.action_type == "delta_joint_angle":
            if action.shape[0] != 6:
                raise ValueError(f"delta_joint_angle action must have shape (6,), got {action.shape}")

            # Держим последнюю целевую позу, а не текущее просевшее положение
            q = self.compute_q + action[:-1]
            q = np.clip(q, q_min, q_max)
            gripper_cmd = np.array([np.clip(action[-1], gripper_min, gripper_max)], dtype=np.float32)

        elif self.action_type == "joint_angle":
            if action.shape[0] != 6:
                raise ValueError(f"joint_angle action must have shape (6,), got {action.shape}")

            q = np.clip(action[:-1], q_min, q_max)
            gripper_cmd = np.array([np.clip(action[-1], gripper_min, gripper_max)], dtype=np.float32)

        else:
            raise ValueError("action_type not recognized")

        self.compute_q = np.asarray(q, dtype=np.float32)
        self.last_q = copy.deepcopy(self.compute_q)
        self.q = np.concatenate([self.compute_q, gripper_cmd]).astype(np.float32)

        if self.state_type == "joint_angle":
            return self.get_joint_state()
        elif self.state_type == "ee_pose":
            return self.get_ee_pose()
        elif self.state_type == "delta_q":
            return self.get_delta_q()
        else:
            raise ValueError("state_type not recognized")

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        """
        Grab images from the environment.

        Returns:
            rgb_agent: RGB с камеры cam_front (на столе, со стороны рабочей зоны)
            rgb_ego:   RGB с камеры cam_side (на столе, сбоку)
        """
        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name=self.agent_cam_name)
        self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name=self.ego_cam_name)
        return self.rgb_agent, self.rgb_ego

    def render(self, teleop=False):
        """Render the environment."""
        if not self._visible_window or not hasattr(self.env, "viewer"):
            return
        if not hasattr(self, "rgb_agent") or not hasattr(self, "rgb_ego"):
            self.grab_image()

        self.env.plot_time()

        rgb_front = add_title_to_img(
            self.rgb_agent, text="Front cam", shape=(640, 480)
        )
        rgb_side = add_title_to_img(
            self.rgb_ego, text="Side cam", shape=(640, 480)
        )

        self.env.viewer_rgb_overlay(rgb_front, loc="top right")
        self.env.viewer_rgb_overlay(rgb_side, loc="bottom right")

        if teleop:
            self.env.viewer_text_overlay(
                text1="Key Pressed", text2=f"{self.env.get_key_pressed_list()}"
            )
            self.env.viewer_text_overlay(
                text1="Key Repeated", text2=f"{self.env.get_key_repeated_list()}"
            )

        self.env.render()

    def get_joint_state(self):
        """
        Get robot joint state:
        [j1,j2,j3,j4,j5,gripper_state]
        """
        qpos = self.env.get_qpos_joints(joint_names=self.arm_joint_names)
        gripper_cmd = self._gripper_cmd_value()
        return np.concatenate([qpos, [gripper_cmd]], dtype=np.float32)

    def teleop_robot(self):
        """
        Convenient joint-space control for SO-101

        Keys:
            Q / A -> shoulder_pan
            W / S -> shoulder_lift
            E / D -> elbow_flex
            I / K -> wrist_flex
            O / L -> wrist_roll
            SPACE -> gripper open/close
            Z -> reset
        """

        def pressed(key):
            return (
                self.env.is_key_pressed_repeat(key=key)
                or self.env.is_key_pressed_once(key=key)
            )

        # fallback for eef control, if ever used
        if self.action_type == "eef_pose":
            dpos = np.zeros(3)
            drot = np.eye(3)

            if pressed(glfw.KEY_S):
                dpos += np.array([0.007, 0.0, 0.0])
            if pressed(glfw.KEY_W):
                dpos += np.array([-0.007, 0.0, 0.0])
            if pressed(glfw.KEY_A):
                dpos += np.array([0.0, -0.007, 0.0])
            if pressed(glfw.KEY_D):
                dpos += np.array([0.0, 0.007, 0.0])
            if pressed(glfw.KEY_R):
                dpos += np.array([0.0, 0.0, 0.007])
            if pressed(glfw.KEY_F):
                dpos += np.array([0.0, 0.0, -0.007])

            if pressed(glfw.KEY_LEFT):
                drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
            if pressed(glfw.KEY_RIGHT):
                drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
            if pressed(glfw.KEY_DOWN):
                drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
            if pressed(glfw.KEY_UP):
                drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
            if pressed(glfw.KEY_Q):
                drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
            if pressed(glfw.KEY_E):
                drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]

            if self.env.is_key_pressed_once(key=glfw.KEY_Z):
                return np.zeros(7, dtype=np.float32), True

            if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
                self.gripper_state = not self.gripper_state

            drot = r2rpy(drot)
            action = np.concatenate(
                [dpos, drot, np.array([self.gripper_state], dtype=np.float32)],
                dtype=np.float32,
            )
            return action, False

        dq = np.zeros(5, dtype=np.float32)

        # Smaller step for finer control
        step = np.deg2rad(1.5)

        # 1) base
        if pressed(glfw.KEY_Q):
            dq[0] += step
        if pressed(glfw.KEY_A):
            dq[0] -= step

        # 2) shoulder
        if pressed(glfw.KEY_W):
            dq[1] += step
        if pressed(glfw.KEY_S):
            dq[1] -= step

        # 3) elbow
        if pressed(glfw.KEY_E):
            dq[2] += step
        if pressed(glfw.KEY_D):
            dq[2] -= step

        # 4) wrist flex
        if pressed(glfw.KEY_I):
            dq[3] += step
        if pressed(glfw.KEY_K):
            dq[3] -= step

        # 5) wrist roll
        if pressed(glfw.KEY_O):
            dq[4] += step
        if pressed(glfw.KEY_L):
            dq[4] -= step

        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(6, dtype=np.float32), True

        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state = not self.gripper_state

        gripper_cmd = 1.2 if self.gripper_state else 0.0

        action = np.concatenate(
            [dq, np.array([gripper_cmd], dtype=np.float32)],
            dtype=np.float32,
        )
        return action, False

    def get_delta_q(self):
        """
        Get delta joint angles:
        [dq1,dq2,dq3,dq4,dq5,gripper_state]
        """
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper_cmd = self._gripper_cmd_value()
        return np.concatenate([delta, [gripper_cmd]], dtype=np.float32)

    def check_success(self):
        """
        Check if the cube is placed on the plate
        and gripper is open and above a height threshold.
        """
        p_cube = self.env.get_p_body("body_obj_cube")
        p_plate = self.env.get_p_body("body_obj_plate_11")
        p_cube_bottom = self.env.get_p_site("bottom_site_cube")
        p_cube_top = self.env.get_p_site("top_site_cube")
        p_plate_top = self.env.get_p_site("top_site_plate_11")
        p_gripper = self.env.get_p_body(self.ee_body_name)
        gr = self._get_gripper_q()

        cube_is_above_plate_xy = np.linalg.norm(p_cube[:2] - p_plate[:2]) < 0.08
        cube_is_on_plate_height = abs(p_cube_bottom[2] - p_plate_top[2]) < 0.03
        gripper_is_open = gr < 0.05
        hand_is_raised = p_gripper[2] > (p_cube_top[2] + 0.08)

        return (
            cube_is_above_plate_xy
            and cube_is_on_plate_height
            and gripper_is_open
            and hand_is_raised
        )

    def get_obj_pose(self):
        """
        Returns:
            cube_pose, plate_pose where each pose is [x, y, z, qw, qx, qy, qz]
        """
        p_cube, R_cube = self.env.get_pR_body("body_obj_cube")
        p_plate, R_plate = self.env.get_pR_body("body_obj_plate_11")
        cube_pose = np.concatenate([p_cube, r2quat(R_cube)], dtype=np.float32)
        plate_pose = np.concatenate([p_plate, r2quat(R_plate)], dtype=np.float32)
        return cube_pose, plate_pose

    def set_obj_pose(self, cube_pose, plate_pose):
        """
        Set object poses.
        """
        cube_pose = np.asarray(cube_pose, dtype=np.float32)
        plate_pose = np.asarray(plate_pose, dtype=np.float32)

        self.env.set_p_base_body(body_name="body_obj_cube", p=cube_pose[:3])
        if cube_pose.shape[0] >= 7:
            self.env.set_R_base_body(body_name="body_obj_cube", R=quat2r(cube_pose[3:7]))
        else:
            self.env.set_R_base_body(body_name="body_obj_cube", R=np.eye(3, 3))

        self.env.set_p_base_body(body_name="body_obj_plate_11", p=plate_pose[:3])
        if plate_pose.shape[0] >= 7:
            self.env.set_R_base_body(body_name="body_obj_plate_11", R=quat2r(plate_pose[3:7]))
        else:
            self.env.set_R_base_body(body_name="body_obj_plate_11", R=np.eye(3, 3))

        self.step_env()

    def get_ee_pose(self):
        """
        Get end-effector pose:
        [px, py, pz, roll, pitch, yaw]
        """
        p, R = self.env.get_pR_body(body_name=self.ee_body_name)
        rpy = r2rpy(R)
        return np.concatenate([p, rpy], dtype=np.float32)
