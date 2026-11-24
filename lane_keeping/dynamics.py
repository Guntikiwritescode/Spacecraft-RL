"""Kinematic bicycle model in a road-aligned frame.

State [s, y, psi, v]: progress along the road [m], lateral offset [m],
heading error [rad] and speed [m/s]. Controls [delta, a]: steering angle [rad]
and acceleration [m/s^2]. The model is

    s' = v cos(psi),  y' = v sin(psi),  psi' = (v / L) tan(delta),  v' = a

integrated with explicit Euler sub-steps. Road curvature is applied in the
environment.
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
            v = max(0.0, v)
            psi = (psi + np.pi) % (2 * np.pi) - np.pi
        return np.array([s, y, psi, v], dtype=np.float64)
