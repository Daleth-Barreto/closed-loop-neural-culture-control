import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[0]
RLGYM = REPO_ROOT / "third_party" / "unitree_rl_gym"
CONFIG = RLGYM / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"


def gravity_orientation(quat):
    qw, qx, qy, qz = quat
    return np.array([2 * (-qz * qx + qw * qy),
                     -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)])


class Deploy12:
    def __init__(self, config_path=CONFIG, headless=True):
        with open(config_path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.cfg = cfg
        xml = RLGYM / cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(RLGYM))
        policy = RLGYM / cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(RLGYM))
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.model.opt.timestep = cfg["simulation_dt"]
        self.data = mujoco.MjData(self.model)
        self.dt = cfg["simulation_dt"]
        self.dec = cfg["control_decimation"]
        self.kps = np.array(cfg["kps"], dtype=float)
        self.kds = np.array(cfg["kds"], dtype=float)
        self.default_angles = np.array(cfg["default_angles"], dtype=float)
        self.n_a = cfg["num_actions"]
        self.n_obs = cfg["num_obs"]
        self.action_scale = cfg["action_scale"]
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=float)
        self.ang_vel_scale = cfg["ang_vel_scale"]
        self.dof_pos_scale = cfg["dof_pos_scale"]
        self.dof_vel_scale = cfg["dof_vel_scale"]
        self.policy = torch.jit.load(str(policy), map_location="cpu").eval()
        self.renderer = None
        self.obs_noise_std = 0.0
        self.cmd_fn = None

    def reset(self, cmd=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.action = np.zeros(self.n_a, dtype=np.float32)
        self.target = self.default_angles.copy()
        if cmd is not None:
            self.cmd = np.asarray(cmd, dtype=float)
        else:
            self.cmd = np.array(self.cfg["cmd_init"], dtype=float)
        self.cmd_fn = None
        self.obs_noise_std = 0.0

    def obs(self):
        d = self.data
        cfg = self.cfg
        obs = np.zeros(self.n_obs, dtype=np.float32)
        obs[0:3] = d.qvel[3:6] * self.ang_vel_scale
        obs[3:6] = gravity_orientation(d.qpos[3:7])
        obs[6:9] = self.cmd * self.cmd_scale
        period = 0.8
        phase = (d.time % period) / period
        obs[9:21] = (d.qpos[7:] - self.default_angles) * self.dof_pos_scale
        obs[21:33] = d.qvel[6:] * self.dof_vel_scale
        obs[33:45] = self.action
        obs[45:47] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
        if self.obs_noise_std > 0:
            obs += self.rng.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)
        return obs

    def refresh_cmd(self):
        if self.cmd_fn is not None:
            self.cmd = np.asarray(self.cmd_fn(self.data.time), dtype=float)

    def analog_pd_tau(self):
        return (self.target - self.data.qpos[7:]) * self.kps + (0.0 - self.data.qvel[6:]) * self.kds

    def step_ctrl(self):
        self.data.ctrl[:] = self.analog_pd_tau()
        mujoco.mj_step(self.model, self.data)

    def apply_ctrl(self, tau):
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

    def policy_step(self):
        with torch.inference_mode():
            out = self.policy(torch.from_numpy(self.obs()).unsqueeze(0))
        self.action = out.detach().numpy().squeeze().astype(np.float32)
        self.target = self.action * self.action_scale + self.default_angles

    def render_frame(self, width=320, height=240):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.renderer.update_scene(self.data)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def run(self, duration, cmd=None, push=None, report_cb=None):
        self.reset(cmd=cmd)
        import time
        t0 = time.time()
        steps = 0
        heights, vels, pitches = [], [], []
        fallen = False
        while self.data.time < duration:
            self.refresh_cmd()
            self.step_ctrl()
            steps += 1
            if steps % self.dec == 0:
                self.policy_step()
            if push is not None and duration * 0.05 <= self.data.time <= duration * 0.05 + push[1]:
                self.data.xfrc_applied[self.model.body("pelvis").id, :3] = np.asarray(push[0])
            else:
                self.data.xfrc_applied[self.model.body("pelvis").id, :3] = 0.0
            if steps % 50 == 0:
                heights.append(self.data.qpos[2])
                vels.append(self.data.qvel[0])
                q = self.data.qpos[3:7]
                pitch = np.arctan2(2 * (q[3]*q[2] + q[0]*q[1]), 1 - 2*(q[1]*q[1] + q[2]*q[2]))
                pitches.append(pitch)
                if report_cb:
                    report_cb(self.data.time, self.data.qpos[2], self.data.qvel[0])
            if self.data.qpos[2] < 0.35:
                fallen = True
                break
        wall = time.time() - t0
        return {
            "sim_time": float(self.data.time),
            "wall_time": float(wall),
            "end_height": float(self.data.qpos[2]),
            "end_vx": float(self.data.qvel[0]),
            "mean_height": float(np.mean(heights)) if heights else float('nan'),
            "min_height": float(np.min(heights)) if heights else float('nan'),
            "max_vx": float(np.max(np.abs(vels))) if vels else float('nan'),
            "mean_vx": float(np.mean(vels)) if vels else float('nan'),
            "mean_pitch": float(np.mean(np.abs(pitches))) if pitches else float('nan'),
            "fallen": fallen,
            "spikes_total": 0,
            "grid": {"t": (heights and self.data.time) if self.data.time else 0, "h": heights, "v": vels},
        }


def main():
    deploy = Deploy12()
    deploy.reset()
    print("joints:", deploy.n_a, "obs:", deploy.n_obs, "policy in:", str(deploy.policy.graph.inputs()))
    res = deploy.run(duration=12.0, cmd=[0.5, 0, 0])
    for k in ("sim_time", "wall_time", "end_height", "mean_height", "min_height", "max_vx", "mean_vx", "mean_pitch", "fallen"):
        print(f"{k}: {res[k]}")


if __name__ == "__main__":
    main()