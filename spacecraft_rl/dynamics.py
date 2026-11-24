"""Clohessy-Wiltshire relative motion in the target-centered Hill frame.

x is radial, y is along-track and z is cross-track. The state is
[x, y, z, vx, vy, vz].
"""

from __future__ import annotations

import numpy as np

MU_EARTH = 3.986004418e14  # Earth's gravitational parameter [m^3 / s^2]
R_EARTH = 6.371e6          # Mean Earth radius [m]


def mean_motion(altitude_m: float) -> float:
    r_orbit = R_EARTH + altitude_m
    return float(np.sqrt(MU_EARTH / r_orbit**3))


class CWDynamics:
    """RK4 propagator with a zero-order hold on the control acceleration."""

    def __init__(self, n: float, substeps: int = 4):
        self.n = float(n)
        self.substeps = int(substeps)

    def _deriv(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
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

    def state_transition_matrix(self, dt: float) -> np.ndarray:
        """Closed-form state-transition matrix of the unforced system."""
        n = self.n
        nt = n * dt
        s, c = np.sin(nt), np.cos(nt)
        phi = np.zeros((6, 6), dtype=np.float64)

        phi[0, 0] = 4.0 - 3.0 * c
        phi[0, 3] = s / n
        phi[0, 4] = 2.0 / n * (1.0 - c)
        phi[1, 0] = 6.0 * (s - nt)
        phi[1, 1] = 1.0
        phi[1, 3] = -2.0 / n * (1.0 - c)
        phi[1, 4] = (4.0 * s - 3.0 * nt) / n
        phi[3, 0] = 3.0 * n * s
        phi[3, 3] = c
        phi[3, 4] = 2.0 * s
        phi[4, 0] = -6.0 * n * (1.0 - c)
        phi[4, 3] = -2.0 * s
        phi[4, 4] = 4.0 * c - 3.0

        phi[2, 2] = c
        phi[2, 5] = s / n
        phi[5, 2] = -n * s
        phi[5, 5] = c
        return phi

    def orbital_period(self) -> float:
        return 2.0 * np.pi / self.n
