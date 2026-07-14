"""Multi-algorithm, multi-seed benchmark of reward hacking.

Trains PPO / SAC / TD3 across seeds and reward architectures, then evaluates
each policy on *ground-truth* mission metrics.  Results are appended to
``results/benchmark_records.jsonl`` (one JSON object per run) so the benchmark
is resumable and multiple invocations accumulate.  Representative evaluation
trajectories (seed 0 of each condition) and per-seed training histories are
saved alongside for the figure pipeline.

Usage
-----
    # core cross-family claim (5 seeds each)
    python -m experiments.benchmark_algorithms \
        --algos PPO SAC TD3 --rewards hackable mitigated \
        --seeds 5 --timesteps 50000

    # PPO mitigation ablation across the whole taxonomy
    python -m experiments.benchmark_algorithms \
        --algos PPO \
        --rewards quota_hackable potential constraint_gated hierarchical \
        --seeds 5 --timesteps 50000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from experiments.common import (RESULTS_DIR, evaluate_agent, make_env,
                                make_model, maybe_init_wandb)
from experiments.callbacks import MetricsCallback

RECORDS_PATH = os.path.join(RESULTS_DIR, "benchmark_records.jsonl")
TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectories")
HISTORY_DIR = os.path.join(RESULTS_DIR, "history")
os.makedirs(TRAJ_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)


def run_one(algo: str, reward_type: str, seed: int, timesteps: int,
            eval_episodes: int, wandb_mode: str = "disabled",
            save_trajectories: bool = False) -> dict:
    """Train one (algo, reward, seed) and return its evaluation record."""
    t0 = time.time()
    env = make_env(reward_type, seed=seed)
    run = None
    if wandb_mode != "disabled":
        run = maybe_init_wandb(
            "goodharts-taxonomy", f"{algo}-{reward_type}-s{seed}",
            dict(algo=algo, reward_type=reward_type, seed=seed,
                 timesteps=timesteps), mode=wandb_mode)

    model = make_model(algo, env, seed=seed, timesteps=timesteps)
    cb = MetricsCallback(wandb_run=run)
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)

    ev = evaluate_agent(model, reward_type, n_episodes=eval_episodes,
                        collect_trajectories=save_trajectories)

    # Save representative trajectories (seed 0) for 3D plots.
    if save_trajectories and ev.get("trajectories"):
        np.savez_compressed(
            os.path.join(TRAJ_DIR, f"{algo}_{reward_type}.npz"),
            *ev["trajectories"])

    # Save training history (compact) for training-curve plots.
    np.savez_compressed(
        os.path.join(HISTORY_DIR, f"{algo}_{reward_type}_s{seed}.npz"),
        dock=np.asarray(cb.docking_successes, dtype=np.int8),
        t=np.asarray(cb.timesteps_at_episode, dtype=np.int64),
        reward=np.asarray(cb.episode_rewards, dtype=np.float32),
    )

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
    with open(RECORDS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    if run is not None:
        run.finish()

    print(f"[{algo:3s} | {reward_type:16s} | seed {seed}] "
          f"dock={record['docking_success_rate']:.0%} "
          f"safe={record['safe_docking_rate']:.0%} "
          f"reward={record['mean_reward']:8.1f} "
          f"final_d={record['mean_final_distance']:6.1f}m "
          f"({record['wall_time_s']:.0f}s)")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
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
    total = len(args.algos) * len(args.rewards) * len(seeds)
    print(f"Benchmark: {total} runs "
          f"({args.algos} x {args.rewards} x {len(seeds)} seeds x "
          f"{args.timesteps} steps)\n")

    done = 0
    for algo in args.algos:
        for reward_type in args.rewards:
            for seed in seeds:
                run_one(algo, reward_type, seed, args.timesteps,
                        args.eval_episodes, wandb_mode=args.wandb,
                        save_trajectories=(seed == seeds[0]))
                done += 1
        print(f"  ... {done}/{total} runs complete\n")

    print(f"Done. Records -> {RECORDS_PATH}")


if __name__ == "__main__":
    main()
