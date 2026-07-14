"""Tests for the docking and lane-keeping environments."""

import numpy as np
import pytest

from stable_baselines3.common.env_checker import check_env

from goodharts_docking import DockingEnv
from goodharts_docking.rewards import REWARD_REGISTRY
from lane_keeping import LaneKeepingEnv
from lane_keeping.rewards import LKREWARD_REGISTRY


@pytest.mark.parametrize("reward_type", list(REWARD_REGISTRY))
def test_docking_env_passes_checker(reward_type):
    check_env(DockingEnv(reward_type=reward_type))


@pytest.mark.parametrize("reward_type", list(LKREWARD_REGISTRY))
def test_lane_env_passes_checker(reward_type):
    check_env(LaneKeepingEnv(reward_type=reward_type))


def test_observation_bounds():
    env = DockingEnv(reward_type="hackable", initial_distance=150)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12,)
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    for _ in range(50):
        obs, *_ = env.step(env.action_space.sample())
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)


def test_reset_determinism():
    e1 = DockingEnv(reward_type="mitigated")
    e2 = DockingEnv(reward_type="mitigated")
    o1, _ = e1.reset(seed=123)
    o2, _ = e2.reset(seed=123)
    assert np.allclose(o1, o2)


def test_info_contains_ground_truth_metrics():
    env = DockingEnv(reward_type="mitigated")
    _, info = env.reset(seed=1)
    for key in ("distance_to_target", "speed", "velocity_violation",
                "cone_violation", "docked", "safe_docked", "lateral_offset",
                "fuel_remaining"):
        assert key in info


def test_pd_controller_can_dock():
    """The environment must be physically solvable (sanity vs. reward hacking)."""
    def pd(info, kp=0.03, kd=1.6):
        p = np.array(info["position"]); v = np.array(info["velocity"])
        return np.clip((-kp * p - kd * v) / DockingEnv.MAX_THRUST, -1, 1)

    env = DockingEnv(reward_type="mitigated", initial_distance=150)
    docks = 0
    for s in range(5):
        obs, info = env.reset(seed=s)
        while True:
            obs, r, term, trunc, info = env.step(pd(info))
            if term or trunc:
                break
        docks += int(info["docked"])
    assert docks >= 4   # a hand controller should dock nearly always


def test_lane_env_solvable():
    def drive(info, target_v=12.0):
        y = info["lateral_offset"]; psi = info["heading_error"]; v = info["speed"]
        kappa = info["curvature"]
        steer = -(0.5 * y + 1.0 * psi) + kappa * 2.8
        accel = 0.5 * (target_v - v)
        return np.clip([steer / LaneKeepingEnv.STEER_MAX,
                        accel / LaneKeepingEnv.ACCEL_MAX], -1, 1)

    env = LaneKeepingEnv(reward_type="mitigated")
    ok = 0
    for s in range(5):
        obs, info = env.reset(seed=s)
        while True:
            obs, r, term, trunc, info = env.step(drive(info))
            if term or trunc:
                break
        ok += int(info["safe_completed"])
    assert ok >= 4
