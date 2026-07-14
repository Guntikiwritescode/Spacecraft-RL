"""Secret mission: transfer the reward-hacking taxonomy to autonomous driving.

Trains PPO on the lane-keeping environment under the same reward-family labels
(hackable=Cobra Effect, quota_hackable=Soviet Quota Gaming, mitigated), then
builds a *cross-domain comparison table* showing that the identical failure
modes and their mitigation recur in a structurally different physical domain --
evidence that the taxonomy is domain-independent rather than orbital-mechanics-
specific.

    python -m experiments.cross_domain --seeds 3 --timesteps 80000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from lane_keeping import LaneKeepingEnv
from experiments.common import RESULTS_DIR, FIGURES_DIR

BENCH_RECORDS = os.path.join(RESULTS_DIR, "benchmark_records.jsonl")
LK_RECORDS = os.path.join(RESULTS_DIR, "lane_keeping_records.jsonl")
LK_TRAJ_DIR = os.path.join(RESULTS_DIR, "lane_trajectories")
os.makedirs(LK_TRAJ_DIR, exist_ok=True)


class LKCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.completions, self.safe_completions = [], []

    def _on_step(self):
        for done, info in zip(self.locals.get("dones", []),
                              self.locals.get("infos", [])):
            if done:
                self.completions.append(bool(info.get("completed", False)))
                self.safe_completions.append(bool(info.get("safe_completed", False)))
        return True


def lk_evaluate(model, reward_type, n_episodes=20, collect_traj=False):
    env = LaneKeepingEnv(reward_type=reward_type)
    rows, trajs = [], []
    for i in range(n_episodes):
        obs, info = env.reset(seed=5000 + i)
        traj = [info["position"]]
        ret = 0.0
        offlane = steps = 0
        max_progress = 0.0
        completed = safe = False
        speeds = []
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            traj.append(info["position"])
            ret += r
            steps += 1
            offlane += int(info["off_lane"])
            speeds.append(info["speed"])
            max_progress = max(max_progress, info["progress"])
            completed = completed or info["completed"]
            safe = safe or info["safe_completed"]
            if term or trunc:
                break
        rows.append({
            "completed": completed, "safe_completed": safe,
            "progress": max_progress, "off_lane_rate": offlane / max(steps, 1),
            "mean_speed": float(np.mean(speeds)), "return": ret,
        })
        if collect_traj:
            trajs.append(np.asarray(traj))

    def agg(k):
        return float(np.mean([r[k] for r in rows]))
    summary = {
        "completion_rate": agg("completed"),
        "safe_completion_rate": agg("safe_completed"),
        "mean_progress": agg("progress"),
        "mean_off_lane_rate": agg("off_lane_rate"),
        "mean_speed": agg("mean_speed"),
        "mean_return": agg("return"),
    }
    return (summary, trajs) if collect_traj else (summary, None)


def lk_train_eval(reward_type, seed, timesteps, eval_episodes):
    t0 = time.time()
    env = LaneKeepingEnv(reward_type=reward_type)
    env.reset(seed=seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, device="cpu",
                learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                policy_kwargs=dict(net_arch=[64, 64]))
    cb = LKCallback()
    model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
    summary, trajs = lk_evaluate(model, reward_type, eval_episodes,
                                 collect_traj=(seed == 0))
    if trajs is not None:
        np.savez_compressed(os.path.join(LK_TRAJ_DIR, f"{reward_type}.npz"), *trajs)
    rec = {"domain": "lane_keeping", "reward_type": reward_type, "seed": seed,
           "timesteps": timesteps, "wall_time_s": round(time.time() - t0, 1),
           **summary}
    with open(LK_RECORDS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[lane {reward_type:15s} seed {seed}] "
          f"complete={rec['completion_rate']:.0%} "
          f"safe={rec['safe_completion_rate']:.0%} "
          f"off_lane={rec['mean_off_lane_rate']:.0%} "
          f"progress={rec['mean_progress']:.0f}m ({rec['wall_time_s']:.0f}s)")
    return rec


def _load(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_cross_domain_figure():
    """Grouped comparison of mission success across the two physical domains."""
    import matplotlib.pyplot as plt
    from experiments.plotting_style import (apply_style, color_for_reward,
                                            label_for_reward, OKABE_ITO)
    apply_style()
    bench = _load(BENCH_RECORDS)
    lk = _load(LK_RECORDS)
    rewards = ["hackable", "quota_hackable", "mitigated"]

    def dock(reward):
        r = [x["safe_docking_rate"] for x in bench
             if x["algo"] == "PPO" and x["reward_type"] == reward]
        return np.mean(r) if r else np.nan

    def lane(reward):
        r = [x["safe_completion_rate"] for x in lk if x["reward_type"] == reward]
        return np.mean(r) if r else np.nan

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(rewards)); w = 0.38
    dvals = [dock(r) for r in rewards]
    lvals = [lane(r) for r in rewards]
    ax.bar(x - w / 2, dvals, w, color=OKABE_ITO["blue"],
           label="Orbital docking (safe-dock rate)", edgecolor="white")
    ax.bar(x + w / 2, lvals, w, color=OKABE_ITO["orange"],
           label="Lane-keeping (safe-completion rate)", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for_reward(r) for r in rewards])
    ax.set_ylabel("Mission success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-domain transfer: same reward families, same outcome\n"
                 "in orbital mechanics and autonomous driving")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig13_cross_domain.png")
    fig.savefig(path); plt.close(fig)
    print(f"  saved {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rewards", nargs="+",
                   default=["hackable", "quota_hackable", "mitigated"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=80_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--figure-only", action="store_true",
                   help="skip training; just (re)build the cross-domain figure")
    args = p.parse_args()
    if not args.figure_only:
        print(f"Cross-domain lane-keeping: {args.rewards} x {args.seeds} seeds "
              f"@ {args.timesteps} steps\n")
        for rt in args.rewards:
            for s in range(args.seeds):
                lk_train_eval(rt, s, args.timesteps, args.eval_episodes)
    build_cross_domain_figure()
    print(f"\nDone. Records -> {LK_RECORDS}")


if __name__ == "__main__":
    main()
