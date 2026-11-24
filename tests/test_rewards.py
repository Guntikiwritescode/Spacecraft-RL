import numpy as np
import pytest

from spacecraft_rl.rewards import (HackableReward, make_reward,
                                       REWARD_REGISTRY, HACKABLE_TYPES,
                                       MITIGATED_TYPES)


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
    r2 = make_reward("hackable", overrides={"not_a_field": 1})
    assert isinstance(r2, HackableReward)


def test_distance_shaping_telescopes():
    """The distance term of the potential reward sums to potential_scale * (d0 - d_T)."""
    r = make_reward("potential")
    r.reset(100.0, 0.0, 0.0)
    path = [100.0, 60.0, 90.0, 50.0]
    total = sum(r.compute(None, None, make_info(d, speed=0.0, lateral=0.0))
                for d in path)
    expected = r.cfg.potential_scale * (100.0 - 50.0)   # telescopes to endpoints
    assert np.isclose(total, expected)


def test_standoff_hold_peaks_off_dock():
    """The standoff bump is largest at standoff_distance, not at the dock."""
    r = make_reward("hackable")
    d_star = r.cfg.standoff_distance
    bump = lambda d: r._standoff_bump(d)
    assert bump(d_star) > bump(1.0)      # more reward at standoff than at dock
    assert bump(d_star) > bump(60.0)     # and than far away
    assert bump(d_star) >= bump(d_star + 5) and bump(d_star) >= bump(d_star - 5)
    assert bump(15.0) > bump(10.0) > bump(5.0)
    assert all(bump(d) >= 0.0 for d in (0.5, 3, 10, 20, 50, 100, 150))


def test_hackable_is_farmable_but_mitigated_is_not():
    """Holding still near the standoff pays under hackable and not under the mitigations."""
    static = make_info(15.0, speed=0.0, lateral=0.0)

    hack = make_reward("hackable"); hack.reset(15.0, 0.0, 0.0)
    assert hack.compute(None, None, static) > 0.5

    for rt in MITIGATED_TYPES:
        m = make_reward(rt); m.reset(15.0, 0.0, 0.0)
        m.compute(None, None, static)    # the first step may pay a one-time phase bonus
        r_static = m.compute(None, None, static)
        assert r_static <= 1e-9, f"{rt} appears farmable: {r_static}"


def test_safe_dock_bonus_dominates_mitigated():
    r = make_reward("mitigated"); r.reset(3.0, 0.2, 0.5)
    docked = make_info(1.0, speed=0.05, lateral=0.1, docked=True, safe_docked=True)
    reward = r.compute(None, None, docked)
    assert reward > 50.0


def test_potential_no_distance_level_reward():
    """Holding position far from the target pays nothing."""
    r = make_reward("potential"); r.reset(80.0, 0.0, 0.0)
    held_far = r.compute(None, None, make_info(80.0, speed=0.0, lateral=0.0))
    assert held_far <= 1e-9
