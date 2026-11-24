"""Make the figures in figures/ from the files in results/.

    python -m experiments.visualize
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from spacecraft_rl import DockingEnv
from experiments.common import (RESULTS_DIR, FIGURES_DIR, bootstrap_ci,
                                group_records, load_jsonl)
from experiments.plotting_style import (apply_style, OKABE_ITO,
                                        color_for_reward,
                                        significance_stars, SEQUENTIAL_CMAP)

TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectories")
RECORDS_PATH = os.path.join(RESULTS_DIR, "benchmark_records.jsonl")
SENS_PATH = os.path.join(RESULTS_DIR, "sensitivity_records.jsonl")
LK_PATH = os.path.join(RESULTS_DIR, "lane_keeping_records.jsonl")
ALGOS = ("PPO", "SAC", "TD3")


def load_trajectories(algo: str, reward: str) -> Optional[List[np.ndarray]]:
    path = os.path.join(TRAJ_DIR, f"{algo}_{reward}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return [data[k] for k in data.files]


def trajectories_available(algos=ALGOS, rewards=("hackable", "mitigated")) -> bool:
    return all(load_trajectories(a, r) is not None for a in algos for r in rewards)


def _save(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, name))
    plt.close(fig)


def fig_trajectories_3d():
    fig = plt.figure(figsize=(5.2 * len(ALGOS), 5.4))
    for i, algo in enumerate(ALGOS):
        ax = fig.add_subplot(1, len(ALGOS), i + 1, projection="3d")
        for reward in ("hackable", "mitigated"):
            c = color_for_reward(reward)
            for j, tr in enumerate(load_trajectories(algo, reward)[:3]):
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, alpha=0.7,
                        lw=1.3, label=reward if j == 0 else None)
                ax.scatter(*tr[-1, :3], color=c, marker="x", s=40)
        ax.scatter([0], [0], [0], color=OKABE_ITO["black"], marker="*", s=120,
                   label="Dock target")
        ax.set_title(algo, fontsize=13)
        ax.set_xlabel("radial x (m)")
        ax.set_ylabel("along-track y (m)")
        ax.set_zlabel("cross-track z (m)")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Evaluation trajectories of the seed 0 policies, first three episodes",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "trajectories_3d.png")


def fig_trajectories_2d():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    cone_deg = DockingEnv.CONE_HALF_ANGLE_DEG
    for ax, reward in zip(axes, ("hackable", "mitigated")):
        c = color_for_reward(reward)
        ax.add_patch(Wedge((0, 0), DockingEnv.CONE_RANGE, -cone_deg, cone_deg,
                           facecolor=OKABE_ITO["sky"], alpha=0.12,
                           label="Approach cone"))
        ax.scatter([0], [0], color=OKABE_ITO["black"], marker="*", s=180,
                   zorder=6, label="Dock target")
        for tr in load_trajectories("PPO", reward)[:5]:
            ax.plot(tr[:, 0], tr[:, 1], color=c, alpha=0.7, lw=1.4)
            ax.scatter(tr[0, 0], tr[0, 1], color=c, marker="o", s=35, zorder=5)
            ax.scatter(tr[-1, 0], tr[-1, 1], color=c, marker="x", s=70, zorder=5)
        ax.set_title(f"PPO, {reward}")
        ax.set_xlabel("radial x (m)")
        ax.set_ylabel("along-track y (m)")
        ax.axhline(0, color="grey", ls="--", alpha=0.3)
        ax.axvline(0, color="grey", ls="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("PPO trajectories of seed 0 in the x-y plane, first five episodes",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "trajectories_2d.png")


def _mean_ci(vals):
    vals = np.asarray(vals, float)
    m = vals.mean()
    lo, hi = bootstrap_ci(vals)
    return m, m - lo, hi - m


def fig_success_bars(groups):
    rewards = ("hackable", "mitigated")
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    width = 0.8 / len(rewards)
    x = np.arange(len(ALGOS))
    for k, reward in enumerate(rewards):
        means, los, his = [], [], []
        for algo in ALGOS:
            m, lo, hi = _mean_ci([r["docking_success_rate"]
                                  for r in groups[(algo, reward)]])
            means.append(m); los.append(lo); his.append(hi)
        ax.bar(x + k * width, means, width, yerr=[los, his], capsize=4,
               color=color_for_reward(reward), label=reward,
               edgecolor="white", linewidth=0.5)
        for xi, m, hi in zip(x + k * width, means, his):
            ax.text(xi, m + hi + 0.015, f"{m:.0%}", ha="center", va="bottom",
                    fontsize=10)
    ax.set_xticks(x + width * (len(rewards) - 1) / 2)
    ax.set_xticklabels(ALGOS, fontsize=12)
    ax.set_ylabel("Docking success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Docking success rate by algorithm and reward\n"
                 "(mean over seeds, bootstrap 95% CI)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "success_bars.png")


def fig_sensitivity_heatmap(sens_records):
    pws = sorted({r["proximity_weight"] for r in sens_records})
    dbs = sorted({r["dock_bonus"] for r in sens_records})
    grid = np.full((len(dbs), len(pws)), np.nan)
    counts = np.zeros_like(grid)
    for r in sens_records:
        i = dbs.index(r["dock_bonus"]); j = pws.index(r["proximity_weight"])
        grid[i, j] = np.nansum([grid[i, j] if not np.isnan(grid[i, j]) else 0,
                                r["docking_success_rate"]])
        counts[i, j] += 1
    with np.errstate(invalid="ignore"):
        grid = grid / np.where(counts == 0, np.nan, counts)

    fig, ax = plt.subplots(figsize=(8, 6.2))
    im = ax.imshow(grid, origin="lower", cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1,
                   aspect="auto")
    ax.set_xticks(range(len(pws))); ax.set_xticklabels([f"{p:g}" for p in pws])
    ax.set_yticks(range(len(dbs))); ax.set_yticklabels([f"{d:g}" for d in dbs])
    ax.set_xlabel("proximity_weight")
    ax.set_ylabel("dock_bonus")
    ax.set_title("PPO docking success rate by reward coefficients\n"
                 "(shaped_hackable reward)")
    for i in range(len(dbs)):
        for j in range(len(pws)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center",
                        color="white" if grid[i, j] < 0.6 else "black", fontsize=9)
    cb = fig.colorbar(im, ax=ax); cb.set_label("Docking success rate")
    fig.tight_layout()
    _save(fig, "sensitivity_heatmap.png")


def fig_mitigations(groups, tests):
    rewards = [r for r in ("hackable", "quota_hackable", "mitigated", "potential",
                           "constraint_gated", "hierarchical")
               if ("PPO", r) in groups]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    data = [[r["docking_success_rate"] for r in groups[("PPO", rw)]] for rw in rewards]
    positions = range(len(rewards))
    rng = np.random.default_rng(0)
    for i, (rw, d) in enumerate(zip(rewards, data)):
        ax.scatter(np.full(len(d), i) + rng.uniform(-0.1, 0.1, len(d)), d,
                   color=color_for_reward(rw), s=50, zorder=5, alpha=0.8,
                   edgecolor="black", linewidth=0.5)
        ax.hlines(np.mean(d), i - 0.3, i + 0.3, color="black", lw=2)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([r for r in rewards], rotation=15, ha="right")
    ax.set_ylabel("Docking success rate")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("PPO docking success rate by reward, one point per seed (line: mean)")
    abl = tests["ppo_mitigation_ablation"]
    for i, rw in enumerate(rewards):
        if rw in abl:
            ax.text(i, 1.07, significance_stars(abl[rw]["p_bonferroni"]),
                    ha="center", fontsize=12)
    fig.tight_layout()
    _save(fig, "mitigations.png")


def fig_constraint_violations(groups):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.arange(len(ALGOS)); width = 0.35
    for ax, key, name in ((axes[0], "mean_velocity_violation_rate", "Velocity corridor"),
                          (axes[1], "mean_cone_violation_rate", "Approach cone")):
        for k, reward in enumerate(("hackable", "mitigated")):
            vals = [np.mean([r[key] for r in groups[(a, reward)]]) for a in ALGOS]
            ax.bar(x + k * width, vals, width, color=color_for_reward(reward),
                   label=reward, edgecolor="white")
        ax.set_xticks(x + width / 2); ax.set_xticklabels(ALGOS)
        ax.set_title(name)
    axes[0].set_ylabel("Fraction of steps in violation")
    axes[0].legend()
    fig.suptitle("Constraint violation rates in evaluation episodes",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "constraint_violations.png")


def fig_cross_domain(bench, lk):
    rewards = ["hackable", "quota_hackable", "mitigated"]

    def mean_of(rows, key, reward):
        return np.mean([r[key] for r in rows if r["reward_type"] == reward])

    ppo = [r for r in bench if r["algo"] == "PPO"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(rewards)); w = 0.38
    series = ((x - w / 2, ppo, "safe_docking_rate", OKABE_ITO["blue"],
               "Docking: safe docking rate (docked without a cone violation)"),
              (x + w / 2, lk, "safe_completion_rate", OKABE_ITO["orange"],
               "Lane keeping: safe completion rate (finished in lane, under the speed limit)"))
    for xs, rows, key, color, label in series:
        vals = [mean_of(rows, key, r) for r in rewards]
        ax.bar(xs, vals, w, color=color, label=label, edgecolor="white")
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", va="bottom",
                    fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([r for r in rewards])
    ax.set_ylabel("Fraction of evaluation episodes")
    ax.set_ylim(0, 1.2)
    ax.set_title("PPO safe success rate by reward, docking and lane keeping")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, "cross_domain.png")


def main():
    apply_style()
    records = load_jsonl(RECORDS_PATH)
    groups = group_records(records)
    with open(os.path.join(RESULTS_DIR, "statistics.json")) as f:
        tests = json.load(f)["tests"]
    if trajectories_available():
        fig_trajectories_3d()
        fig_trajectories_2d()
    else:
        print("results/trajectories/ is missing or incomplete: "
              "skipping trajectories_3d.png and trajectories_2d.png.")
    fig_success_bars(groups)
    fig_sensitivity_heatmap(load_jsonl(SENS_PATH))
    fig_mitigations(groups, tests)
    fig_constraint_violations(groups)
    fig_cross_domain(records, load_jsonl(LK_PATH))


if __name__ == "__main__":
    main()
