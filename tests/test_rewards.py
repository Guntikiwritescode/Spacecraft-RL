"""Property tests for the reward taxonomy (no training required)."""

import numpy as np
import pytest

from goodharts_docking.rewards import (RewardConfig, HackableReward,
                                       MitigatedReward, PotentialBasedReward,
                                       make_reward, REWARD_REGISTRY,
                                       HACKABLE_TYPES, MITIGATED_TYPES)


def make_info(d, speed=0.0, lateral=0.0, docked=False, safe_docked=False,
              vviol=False, cviol=False, fuel_used=0.0, path=0.0):
    phase = 0 if d > 50 else 1 if d > 5 else 2
    return {
        "distance_to_target": d, "speed": speed, "lateral_offset": lateral,
        "fuel_used_this_step": fuel_used, "distance_travelled_this_step": path,
        "velocity_violation": vviol, "cone_violation": cviol,
        "docked": docked, "safe_docked": safe_docked, "phase": phase,
    }


def test_registry_and_factory():
    assert set(REWARD_REGISTRY) == set(HACKABLE_TYPES) | set(MITIGATED_TYPES)
    for rt in REWARD_REGISTRY:
        r = make_reward(rt)
        r.reset(150.0, 0.0, 0.0)
        assert callable(r.compute)
    with pytest.raises(ValueError):
        make_reward("nonexistent")


def test_overrides_applied():
    r = make_reward("hackable", overrides={"proximity_weight": 42.0})
    assert r.cfg.proximity_weight == 42.0
    # unknown keys are ignored, not fatal
    r2 = make_reward("hackable", overrides={"not_a_field": 1})
    assert isinstance(r2, HackableReward)


def test_distance_shaping_telescopes():
    """Potential-based approach shaping is path-independent (a potential diff).

    The distance component of the potential reward telescopes to
    ``potential_scale * (d0 - d_final)`` regardless of the path taken -- the
    'telescoping red herring' that a naive distance-reduction reward cannot, on
    its own, induce farming (Ng et al. 1999)."""
    r = make_reward("potential")
    r.reset(100.0, 0.0, 0.0)
    # Meandering path; speed and lateral held at 0 so only the distance term acts.
    # These far states trigger no dock/violation/fuel terms.
    path = [100.0, 60.0, 90.0, 50.0]
    total = sum(r.compute(None, None, make_info(d, speed=0.0, lateral=0.0))
                for d in path)
    expected = r.cfg.potential_scale * (100.0 - 50.0)   # telescopes to endpoints
    assert np.isclose(total, expected)


def test_standoff_hold_peaks_off_dock():
    """The Cobra-Effect hold bonus is maximal at the standoff distance, not the
    dock -- this is what makes hovering, not docking, the optimum."""
    r = make_reward("hackable")
    d_star = r.cfg.standoff_distance
    bump = lambda d: r._standoff_bump(d)   # station-keeping bonus
    assert bump(d_star) > bump(1.0)      # more reward at standoff than at dock
    assert bump(d_star) > bump(60.0)     # and than far away
    assert bump(d_star) >= bump(d_star + 5) and bump(d_star) >= bump(d_star - 5)
    # monotone decrease from standoff toward the dock (no local max to overshoot)
    assert bump(15.0) > bump(10.0) > bump(5.0)
    # non-negative everywhere (no "give up and diverge" incentive)
    assert all(bump(d) >= 0.0 for d in (0.5, 3, 10, 20, 50, 100, 150))


def test_hackable_is_farmable_but_mitigated_is_not():
    """A static hover state yields positive per-step income under the exploit
    (farmable) but non-positive income under every mitigation (not farmable)."""
    static = make_info(15.0, speed=0.0, lateral=0.0)   # sit at the standoff ring

    hack = make_reward("hackable"); hack.reset(15.0, 0.0, 0.0)
    # prev == current -> telescoping terms are zero; only the hold remains.
    assert hack.compute(None, None, static) > 0.5

    for rt in MITIGATED_TYPES:
        m = make_reward(rt); m.reset(15.0, 0.0, 0.0)
        m.compute(None, None, static)                 # first visit may grant a
        # one-time phase bonus; farmability is about *steady-state* income, so
        # check the second identical hovering step.
        r_static = m.compute(None, None, static)
        assert r_static <= 1e-9, f"{rt} appears farmable: {r_static}"


def test_safe_dock_bonus_dominates_mitigated():
    """Docking safely must pay far more than any single hovering step."""
    r = make_reward("mitigated"); r.reset(3.0, 0.2, 0.5)
    docked = make_info(1.0, speed=0.05, lateral=0.1, docked=True, safe_docked=True)
    reward = r.compute(None, None, docked)
    assert reward > 50.0   # dominated by the +100 sparse safe-dock bonus


def test_potential_no_distance_level_reward():
    """Corrected PBS must not reward merely *being far* (the (1-gamma)*d bug)."""
    r = make_reward("potential"); r.reset(80.0, 0.0, 0.0)
    # Holding position far away: prev == current -> shaping is exactly zero.
    held_far = r.compute(None, None, make_info(80.0, speed=0.0, lateral=0.0))
    assert held_far <= 1e-9
