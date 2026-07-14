"""Hyperparameter / reward-coefficient sensitivity study.

Systematically sweeps the two coefficients that govern the Cobra Effect --
the exploit incentive (``proximity_weight``, the per-tick "hold" bonus) and the
true-objective incentive (``dock_bonus``) -- training a PPO agent at each grid
point and recording the ground-truth docking-success rate.  The resulting
heatmap reveals the *emergence boundary*: the coefficient ratio at which the
reward-maximising policy flips from farming to docking.

Records are appended to ``results/sensitivity_records.jsonl`` (resumable).

Usage
-----
    python -m experiments.sensitivity_study --timesteps 80000 --seeds 2 \
        --init-dist 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from stable_baselines3 import PPO

from goodharts_docking import DockingEnv
from experiments.common import RESULTS_DIR, evaluate_agent

RECORDS_PATH = os.path.join(RESULTS_DIR, "sensitivity_records.jsonl")

# Grid axes (defaults). proximity_weight = exploit strength;
# dock_bonus = reward for actually completing the mission.
PROXIMITY_WEIGHTS = [0.0, 0.5, 1.0, 2.0, 4.0]
DOCK_BONUSES = [10.0, 25.0, 50.0, 100.0, 200.0]


def train_eval_config(proximity_weight: float, dock_bonus: float, seed: int,
                      timesteps: int, init_dist: float,
                      eval_episodes: int) -> dict:
    overrides = {"proximity_weight": proximity_weight, "dock_bonus": dock_bonus}
    env = DockingEnv(reward_type="hackable", reward_overrides=overrides,
                     initial_distance=init_dist)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, device="cpu",
                learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                policy_kwargs=dict(net_arch=[64, 64]))
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    ev = evaluate_agent(model, "hackable", reward_overrides=overrides,
                        n_episodes=eval_episodes, initial_distance=init_dist)
    return {
        "proximity_weight": proximity_weight,
        "dock_bonus": dock_bonus,
        "seed": seed,
        "timesteps": timesteps,
        "init_dist": init_dist,
        "docking_success_rate": ev["docking_success_rate"],
        "safe_docking_rate": ev["safe_docking_rate"],
        "mean_reward": ev["mean_reward"],
        "mean_final_distance": ev["mean_final_distance"],
        "wall_time_s": round(time.time() - t0, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=80_000)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--init-dist", type=float, default=150.0)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--proximity-weights", type=float, nargs="+",
                   default=PROXIMITY_WEIGHTS)
    p.add_argument("--dock-bonuses", type=float, nargs="+", default=DOCK_BONUSES)
    args = p.parse_args()

    grid = [(pw, db, s)
            for pw in args.proximity_weights
            for db in args.dock_bonuses
            for s in range(args.seeds)]
    print(f"Sensitivity grid: {len(args.proximity_weights)} x "
          f"{len(args.dock_bonuses)} x {args.seeds} seeds = {len(grid)} runs "
          f"@ {args.timesteps} steps (init_dist={args.init_dist})\n")

    for i, (pw, db, s) in enumerate(grid, 1):
        rec = train_eval_config(pw, db, s, args.timesteps, args.init_dist,
                                args.eval_episodes)
        with open(RECORDS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i:3d}/{len(grid)}] prox={pw:4.1f} dock_bonus={db:6.1f} "
              f"seed={s} -> dock={rec['docking_success_rate']:.0%} "
              f"final_d={rec['mean_final_distance']:6.1f}m "
              f"({rec['wall_time_s']:.0f}s)")

    print(f"\nDone. Records -> {RECORDS_PATH}")


if __name__ == "__main__":
    main()
