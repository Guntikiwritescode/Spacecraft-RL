"""Grid over proximity_weight and dock_bonus with the shaped_hackable reward.

Trains PPO at each grid point and appends one record per run to
results/sensitivity_records.jsonl.

    python -m experiments.sensitivity_study --timesteps 70000 --seeds 1
"""

from __future__ import annotations

import argparse
import json
import os
import time

from spacecraft_rl import DockingEnv
from experiments.common import (START_DISTANCE, RESULTS_DIR, evaluate_agent,
                                make_model)

RECORDS_PATH = os.path.join(RESULTS_DIR, "sensitivity_records.jsonl")

PROXIMITY_WEIGHTS = [0.0, 8.0, 16.0, 24.0, 32.0]
DOCK_BONUSES = [10.0, 50.0, 100.0, 200.0]


def train_eval_config(proximity_weight: float, dock_bonus: float, seed: int,
                      timesteps: int, init_dist: float,
                      eval_episodes: int) -> dict:
    overrides = {"proximity_weight": proximity_weight, "dock_bonus": dock_bonus}
    env = DockingEnv(reward_type="shaped_hackable", reward_overrides=overrides,
                     initial_distance=init_dist)
    model = make_model("PPO", env, seed=seed)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    ev = evaluate_agent(model, "shaped_hackable", reward_overrides=overrides,
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
    p.add_argument("--timesteps", type=int, default=70_000)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--init-dist", type=float, default=START_DISTANCE)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--proximity-weights", type=float, nargs="+",
                   default=PROXIMITY_WEIGHTS)
    p.add_argument("--dock-bonuses", type=float, nargs="+", default=DOCK_BONUSES)
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for pw in args.proximity_weights:
        for db in args.dock_bonuses:
            for s in range(args.seeds):
                rec = train_eval_config(pw, db, s, args.timesteps,
                                        args.init_dist, args.eval_episodes)
                with open(RECORDS_PATH, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"proximity_weight={pw:g} dock_bonus={db:g} seed={s}: "
                      f"dock={rec['docking_success_rate']:.0%} "
                      f"final_d={rec['mean_final_distance']:.1f}m")


if __name__ == "__main__":
    main()
