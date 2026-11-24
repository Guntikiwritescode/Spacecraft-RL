"""Train PPO on the lane-keeping environment under each reward.

Appends one record per run to results/lane_keeping_records.jsonl.

    python -m experiments.cross_domain --seeds 3 --timesteps 100000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from lane_keeping import LaneKeepingEnv
from experiments.common import RESULTS_DIR, make_model

LK_RECORDS = os.path.join(RESULTS_DIR, "lane_keeping_records.jsonl")


def lk_evaluate(model, reward_type, n_episodes=20):
    env = LaneKeepingEnv(reward_type=reward_type)
    rows = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=5000 + i)
        ret = 0.0
        offlane = steps = 0
        max_progress = 0.0
        completed = safe = False
        speeds = []
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
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

    def agg(k):
        return float(np.mean([r[k] for r in rows]))
    return {
        "completion_rate": agg("completed"),
        "safe_completion_rate": agg("safe_completed"),
        "mean_progress": agg("progress"),
        "mean_off_lane_rate": agg("off_lane_rate"),
        "mean_speed": agg("mean_speed"),
        "mean_return": agg("return"),
    }


def lk_train_eval(reward_type, seed, timesteps, eval_episodes):
    t0 = time.time()
    env = LaneKeepingEnv(reward_type=reward_type)
    env.reset(seed=seed)
    model = make_model("PPO", env, seed=seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    summary = lk_evaluate(model, reward_type, eval_episodes)
    rec = {"domain": "lane_keeping", "reward_type": reward_type, "seed": seed,
           "timesteps": timesteps, "wall_time_s": round(time.time() - t0, 1),
           **summary}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(LK_RECORDS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"lane keeping {reward_type} seed {seed}: "
          f"complete={rec['completion_rate']:.0%} "
          f"safe={rec['safe_completion_rate']:.0%} "
          f"off_lane={rec['mean_off_lane_rate']:.0%} "
          f"progress={rec['mean_progress']:.0f}m")
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rewards", nargs="+",
                   default=["hackable", "quota_hackable", "mitigated"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    args = p.parse_args()
    for rt in args.rewards:
        for s in range(args.seeds):
            lk_train_eval(rt, s, args.timesteps, args.eval_episodes)


if __name__ == "__main__":
    main()
