"""Statistical analysis of the benchmark: hypothesis tests, effect sizes, CIs.

Reads ``results/benchmark_records.jsonl`` and produces:

* ``results/summary_table.csv`` -- per-condition means (docking, safe-docking,
  reward, violations, fuel) with bootstrap 95% CIs.
* ``results/statistics.json``   -- machine-readable test results.
* ``results/statistics_report.txt`` -- a human-readable report.

Design of the tests
--------------------
The **experimental unit is the seed**: each (algo, reward) condition contributes
one docking-success rate per seed.  Between-condition comparisons use the
Mann-Whitney U test on these per-seed rates (non-parametric, no normality
assumption).  With 5 seeds per group and clean separation the two-sided
p-value reaches ~0.008 < 0.01.  Multiple comparisons within a family are
Bonferroni-corrected.  Effect size is reported as the rank-biserial correlation.
Confidence intervals on rates use a seed-level bootstrap.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover
    mannwhitneyu = None

from experiments.common import RESULTS_DIR

RECORDS_PATH = os.path.join(RESULTS_DIR, "benchmark_records.jsonl")

# Deterministic RNG (no wall-clock seeding -> reproducible CIs).
_RNG = np.random.default_rng(12345)


# --------------------------------------------------------------------------- #
# Loading & aggregation
# --------------------------------------------------------------------------- #
def load_records(path: str = RECORDS_PATH) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No benchmark records at {path}. "
                                "Run experiments.benchmark_algorithms first.")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def group_by_condition(records: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in records:
        groups[(r["algo"], r["reward_type"])].append(r)
    return groups


def per_seed_rates(records: List[dict], key: str) -> np.ndarray:
    """One value of ``key`` per seed (records are already per-seed)."""
    return np.array([r[key] for r in records], dtype=float)


# --------------------------------------------------------------------------- #
# Statistics primitives
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000,
                 alpha: float = 0.05) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    idx = _RNG.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def rank_biserial(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation effect size from a Mann-Whitney U statistic."""
    return float(1.0 - 2.0 * u / (n1 * n2))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: probability(a>b) - probability(a<b)."""
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def mann_whitney(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Mann-Whitney U with effect sizes; robust to ties/degeneracy."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = {
        "n1": int(len(a)), "n2": int(len(b)),
        "median_1": float(np.median(a)) if len(a) else float("nan"),
        "median_2": float(np.median(b)) if len(b) else float("nan"),
        "mean_1": float(np.mean(a)) if len(a) else float("nan"),
        "mean_2": float(np.mean(b)) if len(b) else float("nan"),
    }
    if mannwhitneyu is None or len(a) == 0 or len(b) == 0:
        out.update(U=float("nan"), p_value=float("nan"),
                   rank_biserial=float("nan"), cliffs_delta=float("nan"))
        return out
    # If both groups are constant and equal, the test is undefined.
    if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
        out.update(U=float("nan"), p_value=1.0, rank_biserial=0.0,
                   cliffs_delta=0.0)
        return out
    try:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        out.update(U=float("nan"), p_value=1.0, rank_biserial=0.0,
                   cliffs_delta=0.0)
        return out
    out.update(U=float(u), p_value=float(p),
               rank_biserial=rank_biserial(u, len(a), len(b)),
               cliffs_delta=cliffs_delta(a, b))
    return out


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
METRIC_KEYS = [
    "docking_success_rate", "safe_docking_rate", "mean_reward",
    "mean_final_distance", "mean_velocity_violation_rate",
    "mean_cone_violation_rate", "mean_fuel_used",
]

HACKABLE = "hackable"
QUOTA = "quota_hackable"
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

    # (1) Per-algorithm: hackable vs mitigated docking success.
    per_algo = {}
    algos = sorted({a for (a, _) in groups})
    for algo in algos:
        if (algo, HACKABLE) in groups and (algo, "mitigated") in groups:
            a = per_seed_rates(groups[(algo, HACKABLE)], "docking_success_rate")
            b = per_seed_rates(groups[(algo, "mitigated")], "docking_success_rate")
            per_algo[algo] = mann_whitney(a, b)
    tests["per_algorithm_hackable_vs_mitigated"] = per_algo

    # (2) Pooled across algorithms: hackable vs mitigated (the headline claim).
    hack_all = [r for (a, rw), rs in groups.items() if rw == HACKABLE for r in rs]
    mit_all = [r for (a, rw), rs in groups.items() if rw == "mitigated" for r in rs]
    if hack_all and mit_all:
        tests["pooled_hackable_vs_mitigated"] = mann_whitney(
            per_seed_rates(hack_all, "docking_success_rate"),
            per_seed_rates(mit_all, "docking_success_rate"))

    # (3) Mitigation ablation (PPO): hackable vs each mitigation, Bonferroni.
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
            res["n_comparisons"] = n_comp
            ablation[m] = res
    tests["ppo_mitigation_ablation_safe_docking"] = ablation

    return tests


def format_report(summary_rows, tests) -> str:
    L = []
    L.append("=" * 78)
    L.append("GOODHART'S TAXONOMY -- STATISTICAL ANALYSIS")
    L.append("=" * 78)
    L.append("")
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
        return line

    L.append("Mann-Whitney U tests (experimental unit = seed)")
    L.append("-" * 78)
    L.append("(1) Per-algorithm: docking success, hackable vs mitigated")
    for algo, t in tests.get("per_algorithm_hackable_vs_mitigated", {}).items():
        L.append(fmt_test(algo, t))
    if "pooled_hackable_vs_mitigated" in tests:
        L.append("")
        L.append("(2) Pooled across algorithm families (headline claim)")
        L.append(fmt_test("pooled", tests["pooled_hackable_vs_mitigated"]))
    L.append("")
    L.append("(3) PPO mitigation ablation: docking success, hackable vs mitigation")
    L.append("    (Bonferroni-corrected across the mitigation family)")
    for m, t in tests.get("ppo_mitigation_ablation_safe_docking", {}).items():
        L.append(fmt_test(m, t))
    L.append("")
    L.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    L.append("=" * 78)
    return "\n".join(L)


def main():
    records = load_records()
    groups = group_by_condition(records)
    summary_rows = build_summary_table(groups)
    tests = run_hypothesis_tests(groups)

    # Write CSV
    import csv
    csv_path = os.path.join(RESULTS_DIR, "summary_table.csv")
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    with open(os.path.join(RESULTS_DIR, "statistics.json"), "w") as f:
        json.dump({"summary": summary_rows, "tests": tests}, f, indent=2)

    report = format_report(summary_rows, tests)
    with open(os.path.join(RESULTS_DIR, "statistics_report.txt"), "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nWrote: {csv_path}")
    print(f"       {os.path.join(RESULTS_DIR, 'statistics.json')}")
    print(f"       {os.path.join(RESULTS_DIR, 'statistics_report.txt')}")


if __name__ == "__main__":
    main()
