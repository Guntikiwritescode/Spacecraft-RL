import numpy as np

from experiments.common import bootstrap_ci, group_records
from experiments.statistical_analysis import (build_summary_table,
                                              mann_whitney,
                                              run_hypothesis_tests)


def _records(rate_by_reward, algo="PPO", n=5):
    return [{"algo": algo, "reward_type": rw, "seed": s,
             "docking_success_rate": rate, "safe_docking_rate": rate,
             "mean_reward": 0.0, "mean_final_distance": 1.0,
             "mean_velocity_violation_rate": 0.0,
             "mean_cone_violation_rate": 0.0, "mean_fuel_used": 0.0}
            for rw, rate in rate_by_reward.items() for s in range(n)]


def test_equal_constant_groups_have_every_p_value():
    out = mann_whitney(np.zeros(5), np.zeros(5))
    assert out["p_value"] == 1.0
    assert out["p_exact_permutation"] == 1.0
    assert out["rank_biserial"] == 0.0


def test_constant_groups_through_the_ablation():
    rates = {"hackable": 0.0, "mitigated": 0.0, "potential": 0.0,
             "constraint_gated": 0.0, "hierarchical": 0.0}
    groups = group_records(_records(rates))
    tests = run_hypothesis_tests(groups)
    for res in tests["ppo_mitigation_ablation"].values():
        assert res["p_bonferroni"] == 1.0
        assert res["p_exact_bonferroni"] == 1.0
    assert tests["per_algorithm_hackable_vs_mitigated"]["PPO"]["p_value"] == 1.0


def test_different_constant_groups_still_work():
    out = mann_whitney(np.zeros(5), np.ones(5))
    assert out["p_value"] < 0.05
    assert out["p_exact_permutation"] < 0.05


def test_bootstrap_ci_does_not_depend_on_call_order():
    values = np.array([0.0, 0.2, 0.8, 1.0, 0.6])
    first = bootstrap_ci(values)
    bootstrap_ci(np.array([0.1, 0.9, 0.4]))
    assert bootstrap_ci(values) == first


def test_bootstrap_ci_edge_cases():
    assert bootstrap_ci(np.full(5, 0.3)) == (0.3, 0.3)
    assert bootstrap_ci(np.array([0.7])) == (0.7, 0.7)
    lo, hi = bootstrap_ci(np.array([]))
    assert np.isnan(lo) and np.isnan(hi)


def test_summary_table_uses_the_shared_bootstrap():
    groups = group_records(_records({"hackable": 0.0}))
    row = build_summary_table(groups)[0]
    assert row["docking_success_rate_ci_lo"] == bootstrap_ci(np.zeros(5))[0]
