"""Shared publication-quality plotting style.

Applies a consistent, colour-blind-safe design system across every figure:

* Categorical colours use the **Okabe-Ito** palette -- the reference CVD-safe
  categorical set for scientific publishing -- assigned in a *fixed order*
  (never cycled).  Semantic mapping: exploitable rewards -> vermillion (a
  "bad"/warning hue), mitigated rewards -> bluish-green ("good").
* Magnitude (heatmaps) uses ``viridis`` -- perceptually uniform and CVD-safe.
* Recessive grids/spines, legends for >=2 series, thin marks.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Okabe-Ito colour-blind-safe categorical palette (fixed order) ---------- #
OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "grey": "#999999",
}

# Semantic assignment used throughout the paper.
REWARD_COLORS = {
    "hackable": OKABE_ITO["vermillion"],        # exploit / cobra effect
    "quota_hackable": OKABE_ITO["orange"],      # exploit / quota gaming
    "mitigated": OKABE_ITO["green"],            # fixed
    "potential": OKABE_ITO["blue"],
    "constraint_gated": OKABE_ITO["sky"],
    "hierarchical": OKABE_ITO["purple"],
}
REWARD_LABELS = {
    "hackable": "Hackable (Cobra Effect)",
    "quota_hackable": "Quota Gaming",
    "mitigated": "Mitigated",
    "potential": "Potential-Based",
    "constraint_gated": "Constraint-Gated",
    "hierarchical": "Hierarchical",
}
ALGO_COLORS = {
    "PPO": OKABE_ITO["blue"],
    "SAC": OKABE_ITO["orange"],
    "TD3": OKABE_ITO["green"],
}
ALGO_MARKERS = {"PPO": "o", "SAC": "s", "TD3": "^"}

SEQUENTIAL_CMAP = "viridis"


def apply_style():
    """Set global matplotlib rcParams for a clean publication look."""
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def color_for_reward(rt: str) -> str:
    return REWARD_COLORS.get(rt, OKABE_ITO["grey"])


def label_for_reward(rt: str) -> str:
    return REWARD_LABELS.get(rt, rt)


def significance_stars(p: float) -> str:
    if p != p:  # NaN
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
