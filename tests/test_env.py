import numpy as np
import pytest

from stable_baselines3.common.env_checker import check_env

from spacecraft_rl import DockingEnv
from spacecraft_rl.rewards import REWARD_REGISTRY
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


def _reset_at(env, position, velocity=(0.0, 0.0, 0.0), seed=0):
    """Reset, then place the chaser. The phase entry follows the distance."""
    env.reset(seed=seed)
    d = float(np.linalg.norm(position))
    env.state = np.array([*position, *velocity, env.INITIAL_FUEL,
                          env._phase_of(d)], dtype=np.float64)


def test_dock_terminates_episode():
    env = DockingEnv(reward_type="mitigated", initial_distance=100.0)
    _reset_at(env, (1.0, 0.0, 0.0))
    _, _, terminated, truncated, info = env.step(np.zeros(3))
    assert info["docked"] and info["safe_docked"]
    assert terminated and not truncated


def test_fast_pass_inside_dock_radius_is_not_a_dock():
    env = DockingEnv(reward_type="mitigated", initial_distance=100.0)
    _reset_at(env, (1.0, 0.0, 0.0), velocity=(0.0, 0.5, 0.0))
    _, _, terminated, _, info = env.step(np.zeros(3))
    assert not info["docked"] and not terminated


def test_divergence_terminates_with_penalty():
    env = DockingEnv(reward_type="hackable", initial_distance=100.0)
    _reset_at(env, (199.9, 0.0, 0.0), velocity=(1.0, 0.0, 0.0))
    _, reward, terminated, truncated, info = env.step(np.zeros(3))
    assert info["distance_to_target"] > 2.0 * env.INITIAL_DISTANCE
    assert terminated and not truncated
    # the standoff bump is zero this far out, so only the penalty is left
    assert reward == pytest.approx(-env.reward_fn.cfg.divergence_penalty)


def test_time_limit_truncates():
    env = DockingEnv(reward_type="hackable", initial_distance=100.0, max_steps=3)
    env.reset(seed=0)
    for step in range(1, 4):
        _, _, terminated, truncated, _ = env.step(np.zeros(3))
        assert not terminated
        assert truncated == (step == 3)


def test_fuel_exhaustion_stops_thrust():
    env = DockingEnv(reward_type="mitigated", initial_distance=100.0,
                     initial_fuel=0.1)
    env.reset(seed=0)
    full = np.ones(3)
    _, _, _, _, first = env.step(full)
    # the impulse of the step that uses the last fuel is applied in full
    assert first["fuel_used_this_step"] == pytest.approx(0.2 * np.sqrt(3.0))
    assert first["fuel_remaining"] == 0.0
    for _ in range(3):
        _, _, _, _, info = env.step(full)
        assert info["thrust_magnitude"] == 0.0
        assert info["fuel_used_this_step"] == 0.0
        assert info["fuel_remaining"] == 0.0


def test_phase_boundaries_use_named_ranges():
    f = DockingEnv._phase_of
    assert f(DockingEnv.PHASE_CLOSE_RANGE + 0.1) == 0.0
    assert f(DockingEnv.PHASE_CLOSE_RANGE) == 1.0
    assert f(DockingEnv.PHASE_TERMINAL_RANGE + 0.1) == 1.0
    assert f(DockingEnv.PHASE_TERMINAL_RANGE) == 2.0


def test_registered_envs_use_the_environment_step_limit():
    import gymnasium
    import spacecraft_rl  # noqa: F401
    env = gymnasium.make("SpacecraftDocking-hackable-v0", max_steps=7,
                         initial_distance=100.0)
    env.reset(seed=0)
    truncated = False
    for step in range(1, 8):
        _, _, _, truncated, _ = env.step(env.action_space.sample() * 0.0)
        assert truncated == (step == 7)
