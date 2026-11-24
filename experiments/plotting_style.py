"""Colors and matplotlib settings shared by the figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

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

REWARD_COLORS = {
    "hackable": OKABE_ITO["vermillion"],
    "quota_hackable": OKABE_ITO["orange"],
    "mitigated": OKABE_ITO["green"],
    "potential": OKABE_ITO["blue"],
    "constraint_gated": OKABE_ITO["sky"],
    "hierarchical": OKABE_ITO["purple"],
}
ALGO_MARKERS = {"PPO": "o", "SAC": "s", "TD3": "^"}

SEQUENTIAL_CMAP = "viridis"


def apply_style():
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
