"""Train PPO, SAC or TD3 on each reward and evaluate on docking outcomes.

Appends one JSON record per run to results/benchmark_records.jsonl. Evaluation
trajectories of the first seed are saved to results/trajectories.

    python -m experiments.benchmark_algorithms --algos PPO SAC TD3 \
        --rewards hackable mitigated --seeds 5 --timesteps 120000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from experiments.common import (RESULTS_DIR, evaluate_agent, make_env,
                                make_model)
from experiments.callbacks import MetricsCallback

RECORDS_PATH = os.path.join(RESULTS_DIR, "benchmark_records.jsonl")
TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectories")


def run_one(algo: str, reward_type: str, seed: int, timesteps: int,
            eval_episodes: int, wandb_mode: str = "disabled",
            save_trajectories: bool = False) -> dict:
    t0 = time.time()
    env = make_env(reward_type, seed=seed)
    run = None
    if wandb_mode != "disabled":
        import wandb
        run = wandb.init(project="spacecraft-rl",
                         name=f"{algo}-{reward_type}-s{seed}",
                         config=dict(algo=algo, reward_type=reward_type,
                                     seed=seed, timesteps=timesteps),
                         mode=wandb_mode, reinit=True)

    model = make_model(algo, env, seed=seed)
    cb = MetricsCallback(wandb_run=run)
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)

    ev = evaluate_agent(model, reward_type, n_episodes=eval_episodes,
                        collect_trajectories=save_trajectories)

    if save_trajectories and ev.get("trajectories"):
        os.makedirs(TRAJ_DIR, exist_ok=True)
        np.savez_compressed(
            os.path.join(TRAJ_DIR, f"{algo}_{reward_type}.npz"),
            *ev["trajectories"])

    train_sum = cb.summary()
    record = {
        "algo": algo, "reward_type": reward_type, "seed": seed,
        "timesteps": timesteps, "wall_time_s": round(time.time() - t0, 1),
        "train_total_episodes": train_sum["total_episodes"],
        "train_docking_rate": train_sum["docking_success_rate"],
        "train_final_reward": train_sum["final_mean_reward"],
        **{k: v for k, v in ev.items() if k not in ("per_episode", "trajectories")},
        "per_episode": ev["per_episode"],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RECORDS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    if run is not None:
        run.finish()

    print(f"{algo} {reward_type} seed {seed}: "
          f"dock={record['docking_success_rate']:.0%} "
          f"safe={record['safe_docking_rate']:.0%} "
          f"reward={record['mean_reward']:.1f} "
          f"final_d={record['mean_final_distance']:.1f}m")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algos", nargs="+", default=["PPO", "SAC", "TD3"])
    p.add_argument("--rewards", nargs="+", default=["hackable", "mitigated"])
    p.add_argument("--seeds", type=int, default=5,
                   help="number of seeds (0..seeds-1)")
    p.add_argument("--seed-list", type=int, nargs="+", default=None,
                   help="explicit seed list (overrides --seeds)")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--wandb", default="disabled",
                   choices=["disabled", "offline", "online"])
    args = p.parse_args()

    seeds = args.seed_list if args.seed_list is not None else list(range(args.seeds))
    for algo in args.algos:
        for reward_type in args.rewards:
            for seed in seeds:
                run_one(algo, reward_type, seed, args.timesteps,
                        args.eval_episodes, wandb_mode=args.wandb,
                        save_trajectories=(seed == seeds[0]))


if __name__ == "__main__":
    main()
