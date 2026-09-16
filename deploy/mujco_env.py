import yaml
import numpy as np
import mujoco
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT.parent / "third_party"
G1_DEPLOY = THIRD_PARTY / "g1_deploy_mujoco"
MENAGERIE = THIRD_PARTY / "mujoco_menagerie"
G1_DEPLOY_XML = G1_DEPLOY / "g1_xml" / "scene_29dof.xml"
DEFAULT_CONFIG = G1_DEPLOY / "configs" / "g1_29dof_walk.yaml"


def gravity_orientation(quat):
    qw, qx, qy, qz = quat
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


class G1Env:
    def __init__(self, config_path=None, xml_path=None, render_width=640, render_height=480):
        config_path = Path(config_path or DEFAULT_CONFIG)
        with open(config_path, "r") as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
        if xml_path is None:
            xml_path = self._absolute_xml(config_path, G1_DEPLOY_XML)

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = self.cfg["simulation_dt"]
        self.data = mujoco.MjData(self.model)

        self.n_a = self.cfg["num_actions"]
        self.n_obs = self.cfg["num_obs"]
        self.dt = self.cfg["simulation_dt"]
        self.decimation = self.cfg["control_decimation"]
        self.action_scale = self.cfg["action_scale"]
        self.cmd_scale = np.array(self.cfg["cmd_scale"], dtype=float)
        self.ang_vel_scale = self.cfg["ang_vel_scale"]
        self.dof_pos_scale = self.cfg["dof_pos_scale"]
        self.dof_vel_scale = self.cfg["dof_vel_scale"]
        self.kps = np.array(self.cfg["kps"], dtype=float)
        self.kds = np.array(self.cfg["kds"], dtype=float)
        self.policy_joints = self.cfg["policy_joints"]

        xml_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                     for i in range(1, self.model.njnt)]
        self.policy_to_xml = [xml_names.index(n) for n in self.policy_joints]
        self.xml_to_policy = [self.policy_joints.index(n) for n in xml_names]
        assert len(self.policy_to_xml) == self.n_a

        self.default_angles = np.array(self.cfg["default_angles"], dtype=float)[self.policy_to_xml]

        self.renderer = None
        self.render_width = render_width
        self.render_height = render_height

    def _absolute_xml(self, config_path, default_xml_path):
        xml = config_path.parent.parent / self.cfg["xml_path"]
        if not xml.exists():
            xml = default_xml_path
        return Path(xml)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self._frame_stack = [np.zeros(self.n_obs, dtype=float) for _ in range(5)]
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        return self.data

    def _frame(self, cmd, action_xml, default_angles=None):
        if default_angles is None:
            default_angles = self.default_angles
        qj = self.data.qpos[7:] - default_angles
        dqj = self.data.qvel[6:]
        quat = self.data.qpos[3:7]
        f = np.zeros(self.n_obs, dtype=float)
        f[0:3] = self.data.qvel[3:6] * self.ang_vel_scale
        f[3:6] = gravity_orientation(quat)
        f[6:9] = np.asarray(cmd) * self.cmd_scale
        f[9:38] = qj[self.xml_to_policy] * self.dof_pos_scale
        f[38:67] = dqj[self.xml_to_policy] * self.dof_vel_scale
        f[67:96] = np.asarray(action_xml)[self.xml_to_policy]
        return f

    def stacked_obs(self, cmd, action_xml):
        self._frame_stack.append(self._frame(cmd, action_xml))
        self._frame_stack.pop(0)
        frames = np.asarray(self._frame_stack, dtype=float)
        return np.concatenate([
            frames[:, 0:3].ravel(), frames[:, 3:6].ravel(), frames[:, 6:9].ravel(),
            frames[:, 9:38].ravel(), frames[:, 38:67].ravel(), frames[:, 67:96].ravel(),
        ])

    def pd_tau(self, target_dof_pos):
        return (target_dof_pos - self.data.qpos[7:]) * self.kps + (0.0 - self.data.qvel[6:]) * self.kds

    def step_physics(self):
        mujoco.mj_step(self.model, self.data)

    def forward_velocity(self):
        return float(self.data.qvel[0])

    def base_height(self):
        return float(self.data.qpos[2])

    def has_fallen(self, threshold=0.35):
        return self.base_height() < threshold

    def push_torso(self, force, direction=(1.0, 0.0, 0.0), duration=0.1):
        torso_id = self.model.body("torso").id
        start = self.data.time
        while self.data.time - start < duration:
            self.data.xfrc_applied[torso_id, :3] = np.asarray(force) * np.asarray(direction)
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[torso_id, :] = 0.0

    def enable_renderer(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.render_height,
                                            width=self.render_width)
        return self.renderer

    def render_frame(self):
        renderer = self.enable_renderer()
        renderer.update_scene(self.data)
        return renderer.render()

    def release(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None