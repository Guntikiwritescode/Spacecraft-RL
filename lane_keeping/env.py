"""Autonomous lane-keeping / route-completion Gymnasium environment.

A car must drive a gently curving single-lane road from ``s=0`` to a goal
distance, staying within the lane, at a sensible speed.  Like the docking
environment it is *reward-agnostic*: lane and speed constraints are always
tracked in ``info`` but never enforced, so whether the agent drives safely or
exploits the reward is decided entirely by ``reward_type``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium
import numpy as np
from gymnasium import spaces

from lane_keeping.dynamics import LaneDynamics
from lane_keeping.rewards import LKRewardConfig, make_lk_reward


class LaneKeepingEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    DT = 0.1
    MAX_STEPS = 400
    ROUTE_LENGTH = 120.0        # [m] goal distance
    LANE_HALF_WIDTH = 1.75      # [m] (~3.5 m lane)
    OFFROAD_HALF_WIDTH = 4.5    # [m] terminal excursion
    SPEED_LIMIT = 20.0          # [m/s]
    STEER_MAX = 0.5             # [rad]
    ACCEL_MAX = 3.0             # [m/s^2]
    CURV_AMP = 0.02             # [1/m] road curvature amplitude (R ~ 50 m)
    CURV_WAVELENGTH = 60.0      # [m]

    def __init__(self, reward_type: str = "hackable",
                 reward_overrides: Optional[Dict[str, Any]] = None,
                 reward_config: Optional[LKRewardConfig] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self.reward_type = reward_type
        self.dynamics = LaneDynamics()
        self.reward_fn = make_lk_reward(reward_type, reward_config, reward_overrides)

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.state: Optional[np.ndarray] = None    # [s, y, psi, v]
        self.steps = 0
        self.trajectory: list = []

    def _curvature(self, s: float) -> float:
        return self.CURV_AMP * float(np.sin(2 * np.pi * s / self.CURV_WAVELENGTH))

    def _get_obs(self) -> np.ndarray:
        s, y, psi, v = self.state
        obs = np.array([
            y / (2 * self.LANE_HALF_WIDTH),
            psi / (np.pi / 4),
            v / self.SPEED_LIMIT,
            self._curvature(s) / self.CURV_AMP,
            (self.ROUTE_LENGTH - s) / self.ROUTE_LENGTH,
            v * np.sin(psi) / self.SPEED_LIMIT,      # lateral rate
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _get_info(self, s_prev=0.0, y_prev=0.0, path=0.0) -> Dict[str, Any]:
        s, y, psi, v = self.state
        off_lane = bool(abs(y) > self.LANE_HALF_WIDTH)
        off_road = bool(abs(y) > self.OFFROAD_HALF_WIDTH)
        over_speed = bool(v > self.SPEED_LIMIT)
        completed = bool(s >= self.ROUTE_LENGTH)
        safe_completed = bool(completed and not off_lane and not over_speed)
        return {
            "lateral_offset": float(y),
            "heading_error": float(psi),
            "speed": float(v),
            "progress": float(s),
            "progress_this_step": float(s - s_prev),
            "distance_travelled_this_step": float(path),
            "curvature": self._curvature(s),
            "off_lane": off_lane,
            "off_road": off_road,
            "over_speed": over_speed,
            "completed": completed,
            "safe_completed": safe_completed,
            "position": (float(s), float(y)),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        y = self.np_random.uniform(-0.3, 0.3)
        psi = self.np_random.uniform(-0.05, 0.05)
        v = self.np_random.uniform(2.0, 6.0)
        self.state = np.array([0.0, y, psi, v], dtype=np.float64)
        self.steps = 0
        self.trajectory = [(0.0, y)]
        info = self._get_info()
        self.reward_fn.reset(0.0)
        return self._get_obs(), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        s0, y0, psi0, v0 = self.state
        control = np.array([action[0] * self.STEER_MAX, action[1] * self.ACCEL_MAX])
        s, y, psi, v = self.dynamics.step(self.state, control, self.DT)
        v = min(v, self.SPEED_LIMIT * 1.5)   # cap runaway speed

        # Account for road curvature: rotate the heading-error by the road
        # tangent turned through over this step.
        ds = s - s0
        psi -= self._curvature(0.5 * (s + s0)) * ds
        psi = (psi + np.pi) % (2 * np.pi) - np.pi

        path = float(np.hypot(s - s0, y - y0))
        self.state = np.array([s, y, psi, v], dtype=np.float64)
        self.steps += 1
        self.trajectory.append((float(s), float(y)))

        info = self._get_info(s_prev=s0, y_prev=y0, path=path)
        reward = self.reward_fn.compute(info)

        terminated = bool(info["completed"] or info["off_road"])
        truncated = bool(self.steps >= self.MAX_STEPS)
        if info["off_road"]:
            reward -= 20.0
        return self._get_obs(), float(reward), terminated, truncated, info

    def close(self):
        pass
