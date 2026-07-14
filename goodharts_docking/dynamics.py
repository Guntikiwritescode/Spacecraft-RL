"""Full 3D Clohessy-Wiltshire relative orbital dynamics.

The Clohessy-Wiltshire (CW) equations (a.k.a. Hill's equations) linearise the
relative motion of a *chaser* spacecraft about a *target* on a circular
reference orbit.  We work in the target-centred Local-Vertical-Local-Horizontal
(LVLH / Hill) frame:

    x -- radial      (points away from Earth, "downrange" in the plots)
    y -- along-track (direction of orbital velocity, "crosstrack" in-plane)
    z -- cross-track (out of the orbital plane)

The forced equations of motion, with control acceleration ``a = (ax, ay, az)``
and mean motion ``n``, are

    x'' =  3 n^2 x + 2 n y' + ax
    y'' = -2 n x'          + ay
    z'' =       -n^2 z     + az

The in-plane (x, y) coupling produces the counter-intuitive, curved rendezvous
trajectories that make orbital docking hard; the out-of-plane channel (z) is a
decoupled harmonic oscillator ``z'' = -n^2 z`` whose natural period equals the
orbital period, giving the sinusoidal cross-track drift.

This module provides two propagators:

* :meth:`CWDynamics.step` -- an RK4 integrator with sub-stepping.  Treats the
  control acceleration as constant (zero-order hold) over the step.  This is
  what the environment uses each simulation tick.
* :meth:`CWDynamics.state_transition_matrix` -- the closed-form CW STM for the
  unforced system, used for verification/analysis (see ``tests/``).

References
----------
Clohessy, W. H. & Wiltshire, R. S. (1960). *Terminal guidance system for
satellite rendezvous.* Journal of the Aerospace Sciences, 27(9), 653-658.
"""

from __future__ import annotations

import numpy as np

# Physical constants
MU_EARTH = 3.986004418e14  # Earth's gravitational parameter [m^3 / s^2]
R_EARTH = 6.371e6          # Mean Earth radius [m]


def mean_motion(altitude_m: float) -> float:
    """Mean motion ``n`` [rad/s] for a circular orbit at ``altitude_m`` above
    the mean Earth radius."""
    r_orbit = R_EARTH + altitude_m
    return float(np.sqrt(MU_EARTH / r_orbit**3))


class CWDynamics:
    """3D Clohessy-Wiltshire relative-motion propagator.

    Parameters
    ----------
    n:
        Mean motion of the reference orbit [rad/s].
    substeps:
        Number of RK4 sub-steps taken per :meth:`step` call.  With ``dt = 1 s``
        and LEO mean motion (``n ~ 1e-3``) even a single step is extremely
        accurate; the default of 4 keeps trajectories smooth for larger ``dt``.
    """

    def __init__(self, n: float, substeps: int = 4):
        self.n = float(n)
        self.substeps = int(substeps)

    # ------------------------------------------------------------------ #
    # Continuous-time dynamics
    # ------------------------------------------------------------------ #
    def _deriv(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Time derivative of the 6-state ``s = [x, y, z, vx, vy, vz]`` under
        constant control acceleration ``a = [ax, ay, az]``."""
        x, y, z, vx, vy, vz = s
        n = self.n
        ax, ay, az = a
        return np.array(
            [
                vx,
                vy,
                vz,
                3.0 * n * n * x + 2.0 * n * vy + ax,
                -2.0 * n * vx + ay,
                -n * n * z + az,
            ],
            dtype=np.float64,
        )

    def step(self, state6: np.ndarray, accel3: np.ndarray, dt: float) -> np.ndarray:
        """Propagate a 6-state one tick of length ``dt`` under constant control
        acceleration using RK4 with sub-stepping.

        Returns a new ``float64`` array; the input is not mutated.
        """
        s = np.asarray(state6, dtype=np.float64).copy()
        a = np.asarray(accel3, dtype=np.float64)
        h = dt / self.substeps
        for _ in range(self.substeps):
            k1 = self._deriv(s, a)
            k2 = self._deriv(s + 0.5 * h * k1, a)
            k3 = self._deriv(s + 0.5 * h * k2, a)
            k4 = self._deriv(s + h * k3, a)
            s = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return s

    # ------------------------------------------------------------------ #
    # Closed-form state-transition matrix (unforced system)
    # ------------------------------------------------------------------ #
    def state_transition_matrix(self, dt: float) -> np.ndarray:
        """Analytic 6x6 CW state-transition matrix for the *unforced* system.

        Ordering matches the state vector ``[x, y, z, vx, vy, vz]``.  The
        in-plane (x, y) block is the classical CW STM; the out-of-plane block
        is the harmonic-oscillator solution in z.
        """
        n = self.n
        nt = n * dt
        s, c = np.sin(nt), np.cos(nt)
        phi = np.zeros((6, 6), dtype=np.float64)

        # In-plane position rows (x, y) as functions of (x0, y0, vx0, vy0).
        phi[0, 0] = 4.0 - 3.0 * c
        phi[0, 3] = s / n
        phi[0, 4] = 2.0 / n * (1.0 - c)
        phi[1, 0] = 6.0 * (s - nt)
        phi[1, 1] = 1.0
        phi[1, 3] = -2.0 / n * (1.0 - c)
        phi[1, 4] = (4.0 * s - 3.0 * nt) / n
        # In-plane velocity rows (vx, vy).
        phi[3, 0] = 3.0 * n * s
        phi[3, 3] = c
        phi[3, 4] = 2.0 * s
        phi[4, 0] = -6.0 * n * (1.0 - c)
        phi[4, 3] = -2.0 * s
        phi[4, 4] = 4.0 * c - 3.0

        # Out-of-plane harmonic oscillator (z, vz).
        phi[2, 2] = c
        phi[2, 5] = s / n
        phi[5, 2] = -n * s
        phi[5, 5] = c
        return phi

    def orbital_period(self) -> float:
        """Reference orbital period [s] (= out-of-plane oscillation period)."""
        return 2.0 * np.pi / self.n
