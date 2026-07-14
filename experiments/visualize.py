"""Generate the publication-quality figure dashboard from campaign results.

Reads ``results/`` (benchmark records, sensitivity records, saved trajectories
and training histories) and writes 12+ figures to ``figures/``.  Every figure
degrades gracefully: if the data it needs is absent it is skipped with a note,
so the script is useful after a partial campaign.

    python -m experiments.visualize
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from experiments.common import RESULTS_DIR, FIGURES_DIR
from experiments.plotting_style import (apply_style, ALGO_COLORS, ALGO_MARKERS,
                                        REWARD_COLORS, color_for_reward,
                                        label_for_reward, significance_stars,
                                        SEQUENTIAL_CMAP, OKABE_ITO)

apply_style()

TRAJ_DIR = os.path.join(RESULTS_DIR, "trajectories")
HISTORY_DIR = os.path.join(RESULTS_DIR, "history")
RECORDS_PATH = os.path.join(RESULTS_DIR, "benchmark_records.jsonl")
SENS_PATH = os.path.join(RESULTS_DIR, "sensitivity_records.jsonl")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def group_records(records: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    g = defaultdict(list)
    for r in records:
        g[(r["algo"], r["reward_type"])].append(r)
    return g


def load_trajectories(algo: str, reward: str) -> Optional[List[np.ndarray]]:
    path = os.path.join(TRAJ_DIR, f"{algo}_{reward}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return [data[k] for k in data.files]


def load_histories(algo: str, reward: str) -> List[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(HISTORY_DIR, f"{algo}_{reward}_s*.npz"))):
        d = np.load(path)
        out.append({"dock": d["dock"], "t": d["t"], "reward": d["reward"]})
    return out


def _save(fig, name: str):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# 1. 3D trajectory comparison across algorithm families
# --------------------------------------------------------------------------- #
def fig_trajectories_3d(groups):
    algos = [a for a in ("PPO", "SAC", "TD3")
             if load_trajectories(a, "hackable") is not None]
    if not algos:
        print("  [skip] 3D trajectories: no trajectory data")
        return
    n = len(algos)
    fig = plt.figure(figsize=(5.2 * n, 5.4))
    for i, algo in enumerate(algos):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        for reward in ("hackable", "mitigated"):
            trajs = load_trajectories(algo, reward)
            if not trajs:
                continue
            c = color_for_reward(reward)
            for j, tr in enumerate(trajs[:3]):
                ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, alpha=0.7,
                        lw=1.3, label=label_for_reward(reward) if j == 0 else None)
                ax.scatter(*tr[-1, :3], color=c, marker="x", s=40)
        ax.scatter([0], [0], [0], color=OKABE_ITO["black"], marker="*", s=120,
                   label="Dock target")
        ax.set_title(f"{algo}", fontsize=13)
        ax.set_xlabel("radial x (m)")
        ax.set_ylabel("along-track y (m)")
        ax.set_zlabel("cross-track z (m)")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Learned docking trajectories: exploit hovers, mitigation docks "
                 "(3D Clohessy-Wiltshire)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig01_trajectories_3d.png")


# --------------------------------------------------------------------------- #
# 2. 2D projection trajectory comparison (PPO)
# --------------------------------------------------------------------------- #
def fig_trajectories_2d(groups):
    trajs_h = load_trajectories("PPO", "hackable")
    trajs_m = load_trajectories("PPO", "mitigated")
    if trajs_h is None and trajs_m is None:
        print("  [skip] 2D trajectories: no data")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    cone_deg = 20.0
    for ax, trajs, reward in [(axes[0], trajs_h, "hackable"),
                              (axes[1], trajs_m, "mitigated")]:
        c = color_for_reward(reward)
        # approach cone shading (within 50 m of +x axis)
        ax.add_patch(Wedge((0, 0), 50, -cone_deg, cone_deg, facecolor=OKABE_ITO["sky"],
                           alpha=0.12, label="Approach cone"))
        ax.scatter([0], [0], color=OKABE_ITO["black"], marker="*", s=180,
                   zorder=6, label="Dock target")
        if trajs:
            for j, tr in enumerate(trajs[:5]):
                ax.plot(tr[:, 0], tr[:, 1], color=c, alpha=0.7, lw=1.4)
                ax.scatter(tr[0, 0], tr[0, 1], color=c, marker="o", s=35, zorder=5)
                ax.scatter(tr[-1, 0], tr[-1, 1], color=c, marker="x", s=70, zorder=5)
        ax.set_title(f"PPO -- {label_for_reward(reward)}")
        ax.set_xlabel("radial / downrange x (m)")
        ax.set_ylabel("along-track / crosstrack y (m)")
        ax.axhline(0, color="grey", ls="--", alpha=0.3)
        ax.axvline(0, color="grey", ls="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("In-plane trajectories: farming oscillation vs. direct safe dock",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig02_trajectories_2d.png")


# --------------------------------------------------------------------------- #
# 3. Distance-to-target over time
# --------------------------------------------------------------------------- #
def fig_distance_over_time(groups):
    trajs_h = load_trajectories("PPO", "hackable")
    trajs_m = load_trajectories("PPO", "mitigated")
    if not trajs_h and not trajs_m:
        print("  [skip] distance-over-time: no data")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for trajs, reward in [(trajs_h, "hackable"), (trajs_m, "mitigated")]:
        if not trajs:
            continue
        c = color_for_reward(reward)
        for j, tr in enumerate(trajs[:5]):
            d = np.linalg.norm(tr, axis=1)
            ax.plot(d, color=c, alpha=0.6, lw=1.3,
                    label=label_for_reward(reward) if j == 0 else None)
    ax.axhline(1.0, color=OKABE_ITO["black"], ls=":", alpha=0.6,
               label="Dock radius (1 m)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Distance to target (m, symlog)")
    ax.set_title("Distance to target: the exploit stalls near the port, never docking")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig03_distance_over_time.png")


# --------------------------------------------------------------------------- #
# 4. Training curves: docking success over training
# --------------------------------------------------------------------------- #
def _rolling(x, w=25):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return None
    return np.convolve(x, np.ones(w) / w, mode="valid")


def fig_training_curves(groups):
    algos = sorted({a for (a, _) in groups})
    algos = [a for a in ("PPO", "SAC", "TD3") if a in algos]
    if not algos:
        print("  [skip] training curves: no data")
        return
    fig, axes = plt.subplots(1, len(algos), figsize=(5 * len(algos), 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, algo in zip(axes, algos):
        for reward in ("hackable", "mitigated"):
            hists = load_histories(algo, reward)
            if not hists:
                continue
            c = color_for_reward(reward)
            curves = []
            grid = np.linspace(0, 1, 100)
            for h in hists:
                roll = _rolling(h["dock"])
                if roll is None or len(h["t"]) <= 25:
                    continue
                t = h["t"][24:]
                tn = (t - t.min()) / max(t.max() - t.min(), 1)
                curves.append(np.interp(grid, tn, roll))
            if not curves:
                continue
            curves = np.vstack(curves)
            mean = curves.mean(0)
            ax.plot(grid, mean, color=c, lw=2,
                    label=label_for_reward(reward))
            if curves.shape[0] > 1:
                sd = curves.std(0)
                ax.fill_between(grid, mean - sd, mean + sd, color=c, alpha=0.15)
        ax.set_title(algo)
        ax.set_xlabel("Training progress (normalised)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Docking success rate (rolling)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Docking success during training: mitigation learns to dock, "
                 "exploit never does", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig04_training_curves.png")


# --------------------------------------------------------------------------- #
# 5. Headline: docking + safe-docking success across algo x reward
# --------------------------------------------------------------------------- #
def _mean_ci(vals):
    vals = np.asarray(vals, float)
    m = vals.mean()
    if len(vals) < 2:
        return m, 0.0, 0.0
    rng = np.random.default_rng(7)
    boot = vals[rng.integers(0, len(vals), size=(5000, len(vals)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return m, m - lo, hi - m


def fig_success_bars(groups):
    """Headline cross-algorithm figure: the core hackable-vs-mitigated contrast
    across all three algorithm families (the full taxonomy is in fig08)."""
    if not groups:
        print("  [skip] success bars: no data")
        return
    algos = [a for a in ("PPO", "SAC", "TD3") if any(al == a for (al, _) in groups)]
    # Core rewards present for every algorithm family.
    rewards = [r for r in ("hackable", "mitigated")
               if all((a, r) in groups for a in algos)]
    if not rewards:
        rewards = [r for r in ("hackable", "mitigated")
                   if any((a, r) in groups for a in algos)]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    width = 0.8 / max(len(rewards), 1)
    x = np.arange(len(algos))
    for k, reward in enumerate(rewards):
        means, los, his = [], [], []
        for algo in algos:
            recs = groups.get((algo, reward), [])
            if recs:
                m, lo, hi = _mean_ci([r["docking_success_rate"] for r in recs])
            else:
                m, lo, hi = np.nan, 0, 0
            means.append(m); los.append(lo); his.append(hi)
        ax.bar(x + k * width, means, width, yerr=[los, his], capsize=4,
               color=color_for_reward(reward), label=label_for_reward(reward),
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width * (len(rewards) - 1) / 2)
    ax.set_xticklabels(algos, fontsize=12)
    ax.set_ylabel("Docking success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Reward hacking is algorithm-family-independent\n"
                 "(the Cobra-Effect reward collapses docking across PPO, SAC, TD3)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "fig05_success_bars.png")


# --------------------------------------------------------------------------- #
# 6. Goodhart decoupling: reward vs mission success
# --------------------------------------------------------------------------- #
def fig_reward_vs_success(records):
    if not records:
        print("  [skip] reward-vs-success: no data")
        return
    fig, ax = plt.subplots(figsize=(8.5, 6))
    seen = set()
    for r in records:
        rt = r["reward_type"]
        ax.scatter(r["mean_reward"], r["docking_success_rate"],
                   color=color_for_reward(rt),
                   marker=ALGO_MARKERS.get(r["algo"], "o"),
                   s=70, alpha=0.8, edgecolor="white", linewidth=0.6,
                   label=label_for_reward(rt) if rt not in seen else None)
        seen.add(rt)
    # algorithm marker legend (secondary)
    for algo, mk in ALGO_MARKERS.items():
        ax.scatter([], [], color="grey", marker=mk, label=algo)
    ax.set_xlabel("Mean training-reward proxy (per episode)")
    ax.set_ylabel("Ground-truth docking success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Goodhart's Law made visible:\nhigh reward does not imply mission success")
    ax.legend(ncol=2, fontsize=8, loc="center right")
    fig.tight_layout()
    _save(fig, "fig06_reward_vs_success.png")


# --------------------------------------------------------------------------- #
# 7. Sensitivity heatmap
# --------------------------------------------------------------------------- #
def fig_sensitivity_heatmap(sens_records):
    if not sens_records:
        print("  [skip] sensitivity heatmap: no data")
        return
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
    ax.set_xlabel("Exploit incentive: proximity_weight (per-tick hold bonus)")
    ax.set_ylabel("True-objective incentive: dock_bonus")
    ax.set_title("Cobra-Effect emergence boundary\n(PPO docking success vs reward coefficients)")
    for i in range(len(dbs)):
        for j in range(len(pws)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center",
                        color="white" if grid[i, j] < 0.6 else "black", fontsize=9)
    cb = fig.colorbar(im, ax=ax); cb.set_label("Docking success rate")
    fig.tight_layout()
    _save(fig, "fig07_sensitivity_heatmap.png")


# --------------------------------------------------------------------------- #
# 8. Mitigation ablation (PPO) box plot with significance
# --------------------------------------------------------------------------- #
def fig_mitigation_ablation(groups, tests):
    rewards = [r for r in ("hackable", "mitigated", "potential",
                           "constraint_gated", "hierarchical")
               if ("PPO", r) in groups]
    if len(rewards) < 2:
        print("  [skip] mitigation ablation: insufficient data")
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    data = [[r["safe_docking_rate"] for r in groups[("PPO", rw)]] for rw in rewards]
    positions = range(len(rewards))
    bp = ax.boxplot(data, positions=list(positions), widths=0.6,
                    patch_artist=True, showmeans=True)
    for patch, rw in zip(bp["boxes"], rewards):
        patch.set_facecolor(color_for_reward(rw)); patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color("black")
    # scatter individual seeds
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-0.08, 0.08, len(d)),
                   d, color="black", s=18, zorder=5, alpha=0.7)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([label_for_reward(r) for r in rewards], rotation=15, ha="right")
    ax.set_ylabel("Safe-docking rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Mitigation ablation (PPO): each architecture vs the Cobra Effect")
    # significance stars vs hackable
    abl = tests.get("ppo_mitigation_ablation_safe_docking", {}) if tests else {}
    for i, rw in enumerate(rewards):
        if rw in abl:
            p = abl[rw].get("p_bonferroni", abl[rw].get("p_value", float("nan")))
            ax.text(i, 1.0, significance_stars(p), ha="center", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig08_mitigation_ablation.png")


# --------------------------------------------------------------------------- #
# 9. Constraint-violation comparison
# --------------------------------------------------------------------------- #
def fig_constraint_violations(groups):
    algos = [a for a in ("PPO", "SAC", "TD3") if any(al == a for (al, _) in groups)]
    if not algos:
        print("  [skip] constraint violations: no data")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(algos)); width = 0.35
    for k, reward in enumerate(("hackable", "mitigated")):
        vals = []
        for algo in algos:
            recs = groups.get((algo, reward), [])
            vals.append(np.mean([r["mean_velocity_violation_rate"] for r in recs])
                        if recs else np.nan)
        ax.bar(x + k * width, vals, width, color=color_for_reward(reward),
               label=label_for_reward(reward), edgecolor="white")
    ax.set_xticks(x + width / 2); ax.set_xticklabels(algos)
    ax.set_ylabel("Velocity-corridor violation rate")
    ax.set_title("Safety-constraint violations: exploit policies fly unsafely")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig09_constraint_violations.png")


# --------------------------------------------------------------------------- #
# 10. Algorithm-independence box plot
# --------------------------------------------------------------------------- #
def fig_algo_independence(groups):
    algos = [a for a in ("PPO", "SAC", "TD3") if any(al == a for (al, _) in groups)]
    rewards = [r for r in ("hackable", "mitigated") if any(rw == r for (_, rw) in groups)]
    if not algos or not rewards:
        print("  [skip] algo independence: no data")
        return
    fig, ax = plt.subplots(figsize=(9, 5.2))
    positions, data, colors, labels = [], [], [], []
    pos = 0
    tick_pos, tick_lab = [], []
    for reward in rewards:
        block_start = pos
        for algo in algos:
            recs = groups.get((algo, reward), [])
            if not recs:
                continue
            data.append([r["docking_success_rate"] for r in recs])
            positions.append(pos); colors.append(ALGO_COLORS.get(algo, "grey"))
            labels.append(algo); pos += 1
        tick_pos.append((block_start + pos - 1) / 2)
        tick_lab.append(label_for_reward(reward))
        pos += 1
    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    for i, (p, d, c) in enumerate(zip(positions, data, colors)):
        ax.scatter(np.full(len(d), p), d, color="black", s=16, zorder=5, alpha=0.6)
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab)
    ax.set_ylabel("Docking success rate")
    ax.set_ylim(-0.05, 1.05)
    # algo colour legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=ALGO_COLORS[a], label=a, alpha=0.65)
                       for a in algos], title="Algorithm", loc="center left")
    ax.set_title("Same exploit, three algorithm families:\nreward structure, not the optimiser, determines the outcome")
    fig.tight_layout()
    _save(fig, "fig10_algo_independence.png")


# --------------------------------------------------------------------------- #
# 11. Fuel usage
# --------------------------------------------------------------------------- #
def fig_fuel(groups):
    algos = [a for a in ("PPO", "SAC", "TD3") if any(al == a for (al, _) in groups)]
    if not algos:
        print("  [skip] fuel: no data")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(algos)); width = 0.35
    for k, reward in enumerate(("hackable", "mitigated")):
        vals = [np.mean([r["mean_fuel_used"] for r in groups.get((a, reward), [])])
                if groups.get((a, reward)) else np.nan for a in algos]
        ax.bar(x + k * width, vals, width, color=color_for_reward(reward),
               label=label_for_reward(reward), edgecolor="white")
    ax.axhline(50.0, color=OKABE_ITO["black"], ls=":", label="Fuel budget (50 m/s)")
    ax.set_xticks(x + width / 2); ax.set_xticklabels(algos)
    ax.set_ylabel("Mean delta-v used (m/s)")
    ax.set_title("Propellant expenditure by condition")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig11_fuel.png")


# --------------------------------------------------------------------------- #
# 12. Effect-size forest plot
# --------------------------------------------------------------------------- #
def fig_effect_sizes(tests):
    if not tests:
        print("  [skip] effect sizes: no stats")
        return
    rows = []
    for algo, t in tests.get("per_algorithm_hackable_vs_mitigated", {}).items():
        rows.append((f"{algo}: hackable vs mitigated", t.get("rank_biserial", np.nan),
                     t.get("p_value", np.nan)))
    if "pooled_hackable_vs_mitigated" in tests:
        t = tests["pooled_hackable_vs_mitigated"]
        rows.append(("Pooled: hackable vs mitigated",
                     t.get("rank_biserial", np.nan), t.get("p_value", np.nan)))
    for m, t in tests.get("ppo_mitigation_ablation_safe_docking", {}).items():
        rows.append((f"PPO: hackable vs {m}", t.get("rank_biserial", np.nan),
                     t.get("p_bonferroni", t.get("p_value", np.nan))))
    rows = [r for r in rows if not np.isnan(r[1])]
    if not rows:
        print("  [skip] effect sizes: nothing significant to plot")
        return
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 2))
    ys = np.arange(len(rows))
    for y, (label, eff, p) in zip(ys, rows):
        color = OKABE_ITO["green"] if p < 0.05 else OKABE_ITO["grey"]
        ax.scatter(eff, y, color=color, s=90, zorder=5)
        ax.text(eff, y + 0.18, f"{significance_stars(p)}", ha="center", fontsize=10)
    ax.axvline(0, color="grey", ls="--", alpha=0.5)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("Rank-biserial effect size (1 = complete separation)")
    ax.set_title("Effect sizes of reward architecture on mission outcome")
    fig.tight_layout()
    _save(fig, "fig12_effect_sizes.png")


# --------------------------------------------------------------------------- #
def main():
    records = load_jsonl(RECORDS_PATH)
    sens_records = load_jsonl(SENS_PATH)
    groups = group_records(records)

    tests = None
    stats_path = os.path.join(RESULTS_DIR, "statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            tests = json.load(f).get("tests")

    print(f"Loaded {len(records)} benchmark records, "
          f"{len(sens_records)} sensitivity records.")
    print("Generating figures...")
    fig_trajectories_3d(groups)
    fig_trajectories_2d(groups)
    fig_distance_over_time(groups)
    fig_training_curves(groups)
    fig_success_bars(groups)
    fig_reward_vs_success(records)
    fig_sensitivity_heatmap(sens_records)
    fig_mitigation_ablation(groups, tests)
    fig_constraint_violations(groups)
    fig_algo_independence(groups)
    fig_fuel(groups)
    fig_effect_sizes(tests)
    print("Done. Figures in", FIGURES_DIR)


if __name__ == "__main__":
    main()
