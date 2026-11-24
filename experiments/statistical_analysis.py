"""Mann-Whitney tests and bootstrap intervals for results/benchmark_records.jsonl.

Each seed is one observation. Writes results/statistics.json and
results/statistics_report.txt, and prints the report.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from typing import List

import numpy as np

from scipy.stats import mannwhitneyu, rankdata

from experiments.common import (RESULTS_DIR, bootstrap_ci, group_records,
                                load_jsonl)

RECORDS_PATH = os.path.join(RESULTS_DIR, "benchmark_records.jsonl")


def per_seed_rates(records: List[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in records], dtype=float)


def rank_biserial(u: float, n1: int, n2: int) -> float:
    return float(1.0 - 2.0 * u / (n1 * n2))


def exact_permutation_p(a: np.ndarray, b: np.ndarray,
                        max_perms: int = 100_000) -> float:
    """Exact two-sided p value of the rank-sum statistic over all splits of the
    pooled sample into groups of the same sizes. Ties get midranks, so unlike
    SciPy's method="exact" this is valid with tied data. NaN if there are more
    than max_perms splits."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n = len(a), len(a) + len(b)
    if math.comb(n, n1) > max_perms:
        return float("nan")
    ranks = rankdata(np.concatenate([a, b]))
    expected = n1 * (n + 1) / 2.0
    observed = abs(ranks[:n1].sum() - expected)
    hits = total = 0
    for idx in itertools.combinations(range(n), n1):
        total += 1
        hits += abs(ranks[list(idx)].sum() - expected) >= observed - 1e-9
    return hits / total


def mann_whitney(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Mann-Whitney U. Two identical constant samples give p = 1,
    both for the normal approximation and for the exact permutation test."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = {
        "n1": int(len(a)), "n2": int(len(b)),
        "median_1": float(np.median(a)) if len(a) else float("nan"),
        "median_2": float(np.median(b)) if len(b) else float("nan"),
        "mean_1": float(np.mean(a)) if len(a) else float("nan"),
        "mean_2": float(np.mean(b)) if len(b) else float("nan"),
    }
    if len(a) == 0 or len(b) == 0:
        out.update(U=float("nan"), p_value=float("nan"),
                   p_exact_permutation=float("nan"), rank_biserial=float("nan"))
        return out
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        # Every split of the pooled sample gives the same rank sum.
        out.update(U=float("nan"), p_value=1.0, p_exact_permutation=1.0,
                   rank_biserial=0.0)
        return out
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    out.update(U=float(u), p_value=float(p),
               p_exact_permutation=exact_permutation_p(a, b),
               rank_biserial=rank_biserial(u, len(a), len(b)))
    return out


METRIC_KEYS = [
    "docking_success_rate", "safe_docking_rate", "mean_reward",
    "mean_final_distance", "mean_velocity_violation_rate",
    "mean_cone_violation_rate", "mean_fuel_used",
]

HACKABLE = "hackable"
MITIGATIONS = ("mitigated", "potential", "constraint_gated", "hierarchical")


def build_summary_table(groups) -> List[dict]:
    rows = []
    for (algo, reward), recs in sorted(groups.items()):
        row = {"algo": algo, "reward_type": reward, "n_seeds": len(recs)}
        for k in METRIC_KEYS:
            vals = per_seed_rates(recs, k)
            lo, hi = bootstrap_ci(vals)
            row[f"{k}_mean"] = float(np.mean(vals))
            row[f"{k}_ci_lo"] = lo
            row[f"{k}_ci_hi"] = hi
        rows.append(row)
    return rows


def run_hypothesis_tests(groups) -> dict:
    tests = {}

    per_algo = {}
    algos = sorted({a for (a, _) in groups})
    for algo in algos:
        if (algo, HACKABLE) in groups and (algo, "mitigated") in groups:
            a = per_seed_rates(groups[(algo, HACKABLE)], "docking_success_rate")
            b = per_seed_rates(groups[(algo, "mitigated")], "docking_success_rate")
            per_algo[algo] = mann_whitney(a, b)
    tests["per_algorithm_hackable_vs_mitigated"] = per_algo

    hack_all = [r for (a, rw), rs in groups.items() if rw == HACKABLE for r in rs]
    mit_all = [r for (a, rw), rs in groups.items() if rw == "mitigated" for r in rs]
    if hack_all and mit_all:
        tests["pooled_hackable_vs_mitigated"] = mann_whitney(
            per_seed_rates(hack_all, "docking_success_rate"),
            per_seed_rates(mit_all, "docking_success_rate"))

    ablation = {}
    if ("PPO", HACKABLE) in groups:
        base = per_seed_rates(groups[("PPO", HACKABLE)], "docking_success_rate")
        comparisons = [m for m in MITIGATIONS if ("PPO", m) in groups]
        n_comp = max(len(comparisons), 1)
        for m in comparisons:
            b = per_seed_rates(groups[("PPO", m)], "docking_success_rate")
            res = mann_whitney(base, b)
            res["p_bonferroni"] = min(1.0, res["p_value"] * n_comp) \
                if not np.isnan(res["p_value"]) else float("nan")
            res["p_exact_bonferroni"] = min(1.0, res["p_exact_permutation"] * n_comp) \
                if not np.isnan(res["p_exact_permutation"]) else float("nan")
            res["n_comparisons"] = n_comp
            ablation[m] = res
    tests["ppo_mitigation_ablation"] = ablation

    return tests


def format_report(summary_rows, tests) -> str:
    L = []
    L.append("Per-condition summary (mean [95% bootstrap CI over seeds])")
    L.append("-" * 78)
    L.append(f"{'algo':4s} {'reward':17s} {'seeds':>5s} {'dock%':>16s} "
             f"{'safe%':>16s} {'reward':>10s}")
    for r in summary_rows:
        dock = f"{r['docking_success_rate_mean']:.0%} " \
               f"[{r['docking_success_rate_ci_lo']:.0%},{r['docking_success_rate_ci_hi']:.0%}]"
        safe = f"{r['safe_docking_rate_mean']:.0%} " \
               f"[{r['safe_docking_rate_ci_lo']:.0%},{r['safe_docking_rate_ci_hi']:.0%}]"
        L.append(f"{r['algo']:4s} {r['reward_type']:17s} {r['n_seeds']:>5d} "
                 f"{dock:>16s} {safe:>16s} {r['mean_reward_mean']:>10.1f}")
    L.append("")

    def fmt_test(name, t):
        p = t.get("p_value", float("nan"))
        pb = t.get("p_bonferroni")
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        line = (f"  {name:28s} U={t.get('U', float('nan')):>6} "
                f"p={p:.4g} {star}  "
                f"medians {t['median_1']:.2f} vs {t['median_2']:.2f}  "
                f"rank-biserial={t.get('rank_biserial', float('nan')):.2f}")
        if pb is not None:
            line += f"  p_bonf={pb:.4g}"
        pe = t.get("p_exact_permutation")
        if pe is not None and pe == pe:
            line += f"  p_exact={pe:.4g}"
        peb = t.get("p_exact_bonferroni")
        if peb is not None:
            line += f"  p_exact_bonf={peb:.4g}"
        return line

    L.append("Mann-Whitney U tests (experimental unit = seed)")
    L.append("-" * 78)
    L.append("(1) Per-algorithm: docking success, hackable vs mitigated")
    for algo, t in tests.get("per_algorithm_hackable_vs_mitigated", {}).items():
        L.append(fmt_test(algo, t))
    if "pooled_hackable_vs_mitigated" in tests:
        L.append("")
        L.append("(2) Pooled across algorithms: hackable vs mitigated")
        L.append(fmt_test("pooled", tests["pooled_hackable_vs_mitigated"]))
    L.append("")
    L.append("(3) PPO mitigation ablation: docking success, hackable vs mitigation")
    L.append("    (Bonferroni-corrected across the mitigation family)")
    for m, t in tests.get("ppo_mitigation_ablation", {}).items():
        L.append(fmt_test(m, t))
    L.append("")
    L.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    return "\n".join(L)


def main():
    if not os.path.exists(RECORDS_PATH):
        raise FileNotFoundError(f"No benchmark records at {RECORDS_PATH}. "
                                "Run experiments.benchmark_algorithms first.")
    groups = group_records(load_jsonl(RECORDS_PATH))
    summary_rows = build_summary_table(groups)
    tests = run_hypothesis_tests(groups)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "statistics.json"), "w") as f:
        json.dump({"summary": summary_rows, "tests": tests}, f, indent=2)

    report = format_report(summary_rows, tests)
    with open(os.path.join(RESULTS_DIR, "statistics_report.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
