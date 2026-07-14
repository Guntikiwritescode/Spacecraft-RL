"""Kinematic bicycle model in a road-aligned (Frenet) frame.

State ``[s, y, psi, v]``:

* ``s``   -- longitudinal progress along the road centreline [m]
* ``y``   -- lateral offset from the centreline [m] (+ = left)
* ``psi`` -- heading error relative to the road tangent [rad]
* ``v``   -- speed [m/s]

Controls ``[delta, a]``: front-wheel steering angle [rad] and longitudinal
acceleration [m/s^2].  Equations of motion (straight road, curvature kappa=0):

    s'   = v cos(psi)
    y'   = v sin(psi)
    psi' = (v / L) tan(delta)
    v'   = a

Integrated with sub-stepped explicit Euler (the model is smooth and the step is
small, so this is accurate and cheap).
"""

from __future__ import annotations

import numpy as np


class LaneDynamics:
    def __init__(self, wheelbase: float = 2.8, substeps: int = 5):
        self.L = float(wheelbase)
        self.substeps = int(substeps)

    def step(self, state4: np.ndarray, control2: np.ndarray, dt: float) -> np.ndarray:
        s, y, psi, v = [float(x) for x in state4]
        delta, a = float(control2[0]), float(control2[1])
        h = dt / self.substeps
        for _ in range(self.substeps):
            s += h * v * np.cos(psi)
            y += h * v * np.sin(psi)
            psi += h * (v / self.L) * np.tan(delta)
            v += h * a
            v = max(0.0, v)                      # no reverse
            psi = (psi + np.pi) % (2 * np.pi) - np.pi
        return np.array([s, y, psi, v], dtype=np.float64)
