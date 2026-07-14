"""Tests for the 3D Clohessy-Wiltshire dynamics."""

import numpy as np
import pytest

from goodharts_docking.dynamics import CWDynamics, mean_motion, MU_EARTH, R_EARTH


def test_mean_motion_leo():
    n = mean_motion(400_000.0)
    # 400 km LEO mean motion ~ 1.13e-3 rad/s; period ~ 92-93 min.
    assert 1.10e-3 < n < 1.16e-3
    period_min = (2 * np.pi / n) / 60.0
    assert 90 < period_min < 95


def test_rk4_matches_analytic_stm():
    """Unforced RK4 propagation must match the closed-form CW STM."""
    n = mean_motion(400_000.0)
    dyn = CWDynamics(n, substeps=50)
    rng = np.random.default_rng(0)
    for _ in range(20):
        s0 = rng.uniform(-20, 20, size=6)
        for dt in (10.0, 60.0, 300.0):
            s_rk4 = dyn.step(s0, np.zeros(3), dt)
            s_stm = dyn.state_transition_matrix(dt) @ s0
            assert np.allclose(s_rk4, s_stm, atol=1e-6), (dt, s_rk4 - s_stm)


def test_out_of_plane_is_harmonic():
    """z decouples as a harmonic oscillator with the orbital period."""
    n = mean_motion(400_000.0)
    dyn = CWDynamics(n, substeps=20)
    period = dyn.orbital_period()
    s0 = np.array([0, 0, 10.0, 0, 0, 0.0])   # pure z displacement
    s_half = dyn.state_transition_matrix(period / 2) @ s0
    # After half a period the oscillator inverts.
    assert np.isclose(s_half[2], -10.0, atol=1e-3)
    assert abs(s_half[0]) < 1e-6 and abs(s_half[1]) < 1e-6  # stays out of plane


def test_thrust_moves_state():
    n = mean_motion(400_000.0)
    dyn = CWDynamics(n)
    s0 = np.zeros(6)
    s1 = dyn.step(s0, np.array([0.1, 0.0, 0.0]), 1.0)
    assert s1[3] > 0.0   # +x thrust increases vx
    assert s1[0] > 0.0   # and moves +x


def test_input_not_mutated():
    n = mean_motion(400_000.0)
    dyn = CWDynamics(n)
    s0 = np.ones(6)
    _ = dyn.step(s0, np.ones(3) * 0.05, 1.0)
    assert np.array_equal(s0, np.ones(6))
