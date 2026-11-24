"""Spacecraft docking environment with 3D Clohessy-Wiltshire relative motion.

A chaser starts at INITIAL_DISTANCE from the target in a 400 km circular orbit
and docks when it is within 3 m at under 0.3 m/s. Only translation is modeled.
The velocity corridor and the approach cone are reported in ``info`` and are
not enforced. Only the reward can make an agent respect them. The fuel budget
is enforced: thrust is zero once the fuel is used up. The impulse of the step
that uses the last fuel is applied in full, so the budget can be exceeded by
less than one step's impulse (at most 0.35 m/s).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium
import numpy as np
from gymnasium import spaces

from spacecraft_rl.dynamics import CWDynamics, mean_motion
from spacecraft_rl.rewards import RewardConfig, make_reward


class DockingEnv(gymnasium.Env):
    """Docking environment with a 12-D observation and 3-D thrust action.

    ``reward_type`` is a key of ``REWARD_REGISTRY``. ``reward_overrides`` replaces
    ``RewardConfig`` fields. ``initial_distance`` defaults to 500 m.
    """

    DT = 1.0                 # simulation tick [s]
    MAX_STEPS = 500
    MAX_THRUST = 0.2         # per-axis max control accel [m/s^2]
    INITIAL_FUEL = 50.0      # delta-v budget [m/s]
    INITIAL_DISTANCE = 500.0
    DOCK_DISTANCE = 3.0      # soft-capture radius [m]
    DOCK_SPEED = 0.3         # soft-capture speed [m/s]
    CONE_HALF_ANGLE_DEG = 40.0
    CONE_RANGE = 50.0        # cone constraint active within this range [m]
    PHASE_CLOSE_RANGE = 50.0     # phase 0 (far) above this range [m]
    PHASE_TERMINAL_RANGE = 5.0   # phase 2 (terminal) at or below this range [m]
    VEL_SCALE = 3.0          # observation velocity scale [m/s]

    def __init__(
        self,
        reward_type: str = "hackable",
        reward_overrides: Optional[Dict[str, Any]] = None,
        reward_config: Optional[RewardConfig] = None,
        altitude_m: float = 400_000.0,
        initial_distance: Optional[float] = None,
        max_thrust: Optional[float] = None,
        max_steps: Optional[int] = None,
        initial_fuel: Optional[float] = None,
    ):
        super().__init__()
        self.reward_type = reward_type
        self.reward_overrides = reward_overrides

        self.INITIAL_DISTANCE = float(initial_distance) if initial_distance is not None else DockingEnv.INITIAL_DISTANCE
        self.MAX_THRUST = float(max_thrust) if max_thrust is not None else DockingEnv.MAX_THRUST
        self.MAX_STEPS = int(max_steps) if max_steps is not None else DockingEnv.MAX_STEPS
        self.INITIAL_FUEL = float(initial_fuel) if initial_fuel is not None else DockingEnv.INITIAL_FUEL

        self.n = mean_motion(altitude_m)
        self.dynamics = CWDynamics(self.n, substeps=4)
        self.reward_fn = make_reward(reward_type, reward_config, reward_overrides)

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self.state: Optional[np.ndarray] = None  # [x,y,z,vx,vy,vz,fuel,phase]
        self.steps = 0
        self._cone_cos_limit = np.cos(np.radians(self.CONE_HALF_ANGLE_DEG))

    @classmethod
    def _phase_of(cls, distance: float) -> float:
        if distance > cls.PHASE_CLOSE_RANGE:
            return 0.0
        if distance > cls.PHASE_TERMINAL_RANGE:
            return 1.0
        return 2.0

    def _get_obs(self) -> np.ndarray:
        """Order: unit direction, velocity, fuel, phase, coarse range, fine range,
        speed, range rate (positive when moving away). The unit direction keeps
        full resolution near the dock."""
        x, y, z, vx, vy, vz, fuel, phase = self.state
        distance = float(np.sqrt(x * x + y * y + z * z))
        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))
        d0, vs = self.INITIAL_DISTANCE, self.VEL_SCALE
        inv_d = 1.0 / distance if distance > 1e-6 else 0.0
        ux, uy, uz = x * inv_d, y * inv_d, z * inv_d
        range_rate = (x * vx + y * vy + z * vz) * inv_d
        obs = np.array(
            [
                ux, uy, uz,
                vx / vs, vy / vs, vz / vs,
                fuel / self.INITIAL_FUEL,
                phase / 2.0,
                distance / d0,
                np.tanh(distance / 10.0),
                speed / vs,
                range_rate / vs,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def _get_info(self, fuel_used: float = 0.0, dist_travelled: float = 0.0,
                  thrust_mag: float = 0.0) -> Dict[str, Any]:
        x, y, z, vx, vy, vz, fuel, phase = self.state
        distance = float(np.sqrt(x * x + y * y + z * z))
        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))

        max_allowed_speed = max(self.DOCK_SPEED, 0.1 * distance)
        velocity_violation = bool(speed > max_allowed_speed)

        # the cone is centered on the +x axis
        cone_cos = x / distance if distance > 1e-6 else 1.0
        cone_violation = bool(distance < self.CONE_RANGE and cone_cos < self._cone_cos_limit)

        docked = bool(distance < self.DOCK_DISTANCE and speed < self.DOCK_SPEED)
        safe_docked = bool(docked and not cone_violation)
        lateral_offset = float(np.sqrt(y * y + z * z))

        return {
            "distance_to_target": distance,
            "lateral_offset": lateral_offset,
            "speed": speed,
            "fuel_remaining": float(fuel),
            "fuel_used_this_step": float(fuel_used),
            "distance_travelled_this_step": float(dist_travelled),
            "thrust_magnitude": float(thrust_mag),
            "velocity_violation": velocity_violation,
            "cone_violation": cone_violation,
            "docked": docked,
            "safe_docked": safe_docked,
            "phase": int(phase),
            "position": (float(x), float(y), float(z)),
            "velocity": (float(vx), float(vy), float(vz)),
            "range_rate": float((x * vx + y * vy + z * vz) / distance) if distance > 1e-6 else 0.0,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """``options`` is part of the Gymnasium signature and is not used."""
        super().reset(seed=seed)
        d0 = self.INITIAL_DISTANCE
        az = self.np_random.uniform(-0.3, 0.3)
        el = self.np_random.uniform(-0.2, 0.2)
        x = d0 * np.cos(el) * np.cos(az)
        y = d0 * np.cos(el) * np.sin(az)
        z = d0 * np.sin(el)
        vx = self.np_random.uniform(-0.10, 0.0)
        vy = self.np_random.uniform(-0.05, 0.05)
        vz = self.np_random.uniform(-0.05, 0.05)

        distance = float(np.sqrt(x * x + y * y + z * z))
        self.state = np.array([x, y, z, vx, vy, vz, self.INITIAL_FUEL,
                               self._phase_of(distance)], dtype=np.float64)
        self.steps = 0

        info = self._get_info()
        self.reward_fn.reset(info["distance_to_target"], info["speed"],
                             info["lateral_offset"])
        return self._get_obs(), info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        x, y, z, vx, vy, vz, fuel, _ = self.state

        thrust = action * self.MAX_THRUST
        thrust_mag = float(np.linalg.norm(thrust))
        if fuel <= 0.0:
            thrust = np.zeros(3)
            thrust_mag = 0.0
        fuel_used = thrust_mag * self.DT
        fuel = max(0.0, fuel - fuel_used)

        s6 = np.array([x, y, z, vx, vy, vz], dtype=np.float64)
        s6_new = self.dynamics.step(s6, thrust, self.DT)
        xn, yn, zn, vxn, vyn, vzn = s6_new

        distance = float(np.sqrt(xn * xn + yn * yn + zn * zn))
        dist_travelled = float(np.linalg.norm(s6_new[:3] - s6[:3]))
        phase = self._phase_of(distance)

        self.state = np.array([xn, yn, zn, vxn, vyn, vzn, fuel, phase], dtype=np.float64)
        self.steps += 1

        info = self._get_info(fuel_used=fuel_used, dist_travelled=dist_travelled,
                              thrust_mag=thrust_mag)
        reward = self.reward_fn.compute(self.state, action, info)

        terminated = bool(info["docked"])
        truncated = bool(self.steps >= self.MAX_STEPS)

        if distance > self.INITIAL_DISTANCE * 2.0:
            terminated = True
            reward -= self.reward_fn.cfg.divergence_penalty

        return self._get_obs(), float(reward), terminated, truncated, info
