"""Training and evaluation code shared by the experiment scripts."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from spacecraft_rl import DockingEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")

# Initial range of all training and evaluation episodes. The environment default is 500 m.
START_DISTANCE = 100.0


def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def group_records(records: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    """Group benchmark records by (algo, reward_type)."""
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in records:
        groups[(r["algo"], r["reward_type"])].append(r)
    return groups


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 12345) -> Tuple[float, float]:
    """Percentile bootstrap interval of the mean. Each call draws from its own
    generator seeded with ``seed``, so the result does not depend on call order."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def make_env(reward_type: str = "hackable", seed: Optional[int] = None,
             initial_distance: float = START_DISTANCE):
    env = DockingEnv(reward_type=reward_type, initial_distance=initial_distance)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return Monitor(env)


def make_model(algo: str, env, seed: int):
    """PPO, SAC or TD3 with the same discount, learning rate and [64, 64] network."""
    algo = algo.upper()
    common = dict(policy="MlpPolicy", env=env, seed=seed, device="cpu",
                  verbose=0, gamma=0.99, learning_rate=3e-4)
    net = [64, 64]

    if algo == "PPO":
        return PPO(**common, n_steps=2048, batch_size=64, n_epochs=10,
                   gae_lambda=0.95, clip_range=0.2,
                   policy_kwargs=dict(net_arch=net))
    if algo == "SAC":
        return SAC(**common, buffer_size=100_000, batch_size=256, tau=0.005,
                   train_freq=1, gradient_steps=1, learning_starts=1_000,
                   policy_kwargs=dict(net_arch=net))
    if algo == "TD3":
        n_actions = env.action_space.shape[-1]
        noise = NormalActionNoise(mean=np.zeros(n_actions),
                                  sigma=0.1 * np.ones(n_actions))
        return TD3(**common, buffer_size=100_000, batch_size=256, tau=0.005,
                   train_freq=1, gradient_steps=1, learning_starts=1_000,
                   action_noise=noise, policy_kwargs=dict(net_arch=net))
    raise ValueError(f"Unknown algo {algo!r}; choose PPO, SAC or TD3.")


def evaluate_agent(model, reward_type: str,
                   reward_overrides: Optional[Dict[str, Any]] = None,
                   n_episodes: int = 20, seed_offset: int = 10_000,
                   collect_trajectories: bool = False,
                   initial_distance: float = START_DISTANCE) -> Dict[str, Any]:
    """Run deterministic episodes. Docking, constraint and fuel metrics are read
    from the environment's ``info``. The training reward is reported separately
    as ``mean_reward``."""
    env = DockingEnv(reward_type=reward_type, reward_overrides=reward_overrides,
                     initial_distance=initial_distance)
    per_ep: List[Dict[str, Any]] = []
    trajectories: List[np.ndarray] = []

    for i in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + i)
        traj = [info["position"]]
        ep_reward = 0.0
        vviol = cviol = steps = 0
        min_dist = info["distance_to_target"]
        docked = safe_docked = False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            traj.append(info["position"])
            ep_reward += reward
            steps += 1
            vviol += int(info["velocity_violation"])
            cviol += int(info["cone_violation"])
            min_dist = min(min_dist, info["distance_to_target"])
            docked = docked or info["docked"]
            safe_docked = safe_docked or info["safe_docked"]
            if terminated or truncated:
                break
        per_ep.append({
            "docked": bool(docked),
            "safe_docked": bool(safe_docked),
            "reward": float(ep_reward),
            "episode_length": int(steps),
            "final_distance": float(info["distance_to_target"]),
            "min_distance": float(min_dist),
            "final_speed": float(info["speed"]),
            "velocity_violation_rate": vviol / max(steps, 1),
            "cone_violation_rate": cviol / max(steps, 1),
            "fuel_used": float(env.INITIAL_FUEL - info["fuel_remaining"]),
        })
        if collect_trajectories:
            trajectories.append(np.asarray(traj, dtype=float))

    def agg(key):
        return float(np.mean([e[key] for e in per_ep]))

    summary = {
        "n_episodes": n_episodes,
        "docking_success_rate": agg("docked"),
        "safe_docking_rate": agg("safe_docked"),
        "mean_reward": agg("reward"),
        "std_reward": float(np.std([e["reward"] for e in per_ep])),
        "mean_episode_length": agg("episode_length"),
        "mean_final_distance": agg("final_distance"),
        "mean_min_distance": agg("min_distance"),
        "mean_final_speed": agg("final_speed"),
        "mean_velocity_violation_rate": agg("velocity_violation_rate"),
        "mean_cone_violation_rate": agg("cone_violation_rate"),
        "mean_fuel_used": agg("fuel_used"),
        "per_episode": per_ep,
    }
    if collect_trajectories:
        summary["trajectories"] = trajectories
    return summary
