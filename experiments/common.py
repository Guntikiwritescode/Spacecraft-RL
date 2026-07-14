"""Shared utilities for the reward-hacking campaign.

Centralises: algorithm construction (PPO / SAC / TD3 with matched
hyperparameters), environment factories, detailed deterministic evaluation, and
optional Weights & Biases tracking.  Keeping these in one place guarantees that
every experiment (single-run, benchmark, sensitivity) uses *identical* training
and evaluation code, so cross-condition comparisons are fair.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from goodharts_docking import DockingEnv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "results", "models")
FIGURES_DIR = os.path.join(ROOT, "figures")

for _d in (RESULTS_DIR, MODELS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# The environment supports the full proximity-operations range (default 500 m),
# but the multi-algorithm campaign trains in the terminal-rendezvous regime
# (150 m) so that all three families learn a docking policy within a CPU-
# feasible budget. This is documented in the paper's methodology.
CAMPAIGN_DISTANCE = 100.0


# --------------------------------------------------------------------------- #
# Environment factory
# --------------------------------------------------------------------------- #
def make_env(reward_type: str = "hackable",
             reward_overrides: Optional[Dict[str, Any]] = None,
             seed: Optional[int] = None,
             monitor: bool = True,
             initial_distance: float = CAMPAIGN_DISTANCE) -> DockingEnv:
    """Construct a (optionally Monitor-wrapped) docking environment."""
    env = DockingEnv(reward_type=reward_type, reward_overrides=reward_overrides,
                     initial_distance=initial_distance)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    if monitor:
        env = Monitor(env)
    return env


# --------------------------------------------------------------------------- #
# Algorithm factory -- matched hyperparameters across families
# --------------------------------------------------------------------------- #
def make_model(algo: str, env, seed: int, timesteps: int = 100_000,
               tensorboard_log: Optional[str] = None):
    """Build an SB3 model of family ``algo`` with fixed, comparable settings.

    On-policy (PPO) and off-policy (SAC/TD3) families necessarily differ in some
    hyperparameters, but discount, network width, learning rate and device are
    held constant so behavioural differences reflect the algorithm family, not
    tuning.
    """
    algo = algo.upper()
    common = dict(policy="MlpPolicy", env=env, seed=seed, device="cpu",
                  verbose=0, gamma=0.99, learning_rate=3e-4,
                  tensorboard_log=tensorboard_log)
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


ALGOS = ("PPO", "SAC", "TD3")


# --------------------------------------------------------------------------- #
# Detailed deterministic evaluation
# --------------------------------------------------------------------------- #
def evaluate_agent(model, reward_type: str,
                   reward_overrides: Optional[Dict[str, Any]] = None,
                   n_episodes: int = 20, seed_offset: int = 10_000,
                   collect_trajectories: bool = False,
                   initial_distance: float = CAMPAIGN_DISTANCE) -> Dict[str, Any]:
    """Run ``n_episodes`` deterministic evaluation episodes and aggregate a rich
    set of behavioural metrics.

    The metrics are *ground-truth* mission outcomes (docking, safe docking,
    constraint violations, fuel), computed independently of the training reward
    -- this is what exposes reward hacking.
    """
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
            "fuel_used": float(DockingEnv.INITIAL_FUEL - info["fuel_remaining"]),
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


# --------------------------------------------------------------------------- #
# Optional Weights & Biases integration (offline by default)
# --------------------------------------------------------------------------- #
def maybe_init_wandb(project: str, name: str, config: Dict[str, Any],
                     mode: Optional[str] = None):
    """Initialise a W&B run if wandb is importable.

    Defaults to ``WANDB_MODE`` env var, else ``"offline"`` so the campaign is
    fully runnable without credentials.  Returns the run handle or ``None``.
    """
    try:
        import wandb
    except Exception:
        return None
    mode = mode or os.environ.get("WANDB_MODE", "offline")
    try:
        return wandb.init(project=project, name=name, config=config,
                          mode=mode, reinit=True)
    except Exception:
        return None
