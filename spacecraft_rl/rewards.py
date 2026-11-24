"""Reward functions for the docking environment.

Each class implements ``reset(distance, speed, lateral)`` and
``compute(state, action, info)``. ``hackable``, ``quota_hackable`` and
``shaped_hackable`` pay for staying near the target and do not see the velocity
corridor or the approach cone. The other four are the mitigations, and they
pay for both. ``RewardConfig`` holds every coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RewardConfig:
    # approach shaping: coefficient on the distance, speed and lateral terms
    shaping_weight: float = 3.0

    # standoff bump of the hackable rewards
    proximity_weight: float = 3.0
    standoff_distance: float = 20.0
    reach_far: float = 120.0
    reach_near: float = 15.0

    activity_weight: float = 3.0

    dock_bonus: float = 5.0
    dock_bonus_mitigated: float = 100.0

    velocity_penalty: float = 5.0
    cone_penalty: float = 2.0
    gate_closed: float = 0.3

    speed_shaping_weight: float = 8.0
    lateral_shaping_weight: float = 12.0

    fuel_weight: float = 0.01
    fuel_weight_mitigated: float = 0.05

    # subtracted by the environment when the episode ends by divergence
    divergence_penalty: float = 50.0

    potential_scale: float = 3.0

    def merged(self, overrides: Optional[Dict[str, Any]]) -> "RewardConfig":
        """Return a copy with ``overrides`` applied; unknown keys are ignored."""
        if not overrides:
            return self
        known = {k: v for k, v in overrides.items() if k in self.__dataclass_fields__}
        return replace(self, **known)


class RewardFunction:
    def __init__(self, config: Optional[RewardConfig] = None):
        self.cfg = config or RewardConfig()
        self.prev_distance: Optional[float] = None
        self.prev_speed: float = 0.0
        self.prev_lateral: float = 0.0

    def reset(self, initial_distance: float, initial_speed: float = 0.0,
              initial_lateral: float = 0.0) -> None:
        self.prev_distance = float(initial_distance)
        self.prev_speed = float(initial_speed)
        self.prev_lateral = float(initial_lateral)

    def compute(self, state, action, info) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def _fuel_term(self, info, weight: float) -> float:
        return -weight * info["fuel_used_this_step"]

    def _standoff_bump(self, d: float) -> float:
        """proximity_weight * max(0, f). f is 1 at standoff_distance, falls
        linearly to 0 over reach_far beyond it and over reach_near inside it."""
        cfg = self.cfg
        if d >= cfg.standoff_distance:
            frac = 1.0 - (d - cfg.standoff_distance) / cfg.reach_far
        else:
            frac = 1.0 - (cfg.standoff_distance - d) / cfg.reach_near
        return cfg.proximity_weight * max(0.0, frac)

    def _approach_shaping(self, info, coef: float) -> float:
        """coef * (decrease in distance + speed_shaping_weight * decrease in speed
        + lateral_shaping_weight * decrease in lateral offset) since the last step.
        Summed over an episode it depends only on the first and last states."""
        cfg = self.cfg
        d = info["distance_to_target"]
        speed = info["speed"]
        lateral = info["lateral_offset"]
        return coef * ((self.prev_distance - d)
                       + cfg.speed_shaping_weight * (self.prev_speed - speed)
                       + cfg.lateral_shaping_weight * (self.prev_lateral - lateral))

    def _update_prev(self, info) -> None:
        self.prev_distance = info["distance_to_target"]
        self.prev_speed = info["speed"]
        self.prev_lateral = info["lateral_offset"]


class HackableReward(RewardFunction):
    """r = standoff_bump(d) - fuel_weight * fuel_used + dock_bonus * docked

    The bump is largest at the standoff distance and zero below 5 m, and docking
    ends the episode, so holding near the standoff pays more than docking.
    """

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        d = info["distance_to_target"]
        r = self._standoff_bump(d)
        r += self._fuel_term(info, cfg.fuel_weight)
        if info["docked"]:
            r += cfg.dock_bonus
        return float(r)


class QuotaHackableReward(RewardFunction):
    """r = standoff_bump(d) + activity_weight * thrust_magnitude
          - fuel_weight * fuel_used + dock_bonus * docked

    The thrust term pays for control effort whether or not it brings the
    spacecraft closer.
    """

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        d = info["distance_to_target"]
        r = self._standoff_bump(d)
        r += cfg.activity_weight * info["thrust_magnitude"]
        r += self._fuel_term(info, cfg.fuel_weight)
        if info["docked"]:
            r += cfg.dock_bonus
        return float(r)


class ShapedHackableReward(RewardFunction):
    """Reward of the sensitivity study.

    r = 2 * (decrease in distance) + proximity_weight * bump(d) + dock_bonus * docked
        - fuel_weight_mitigated * fuel_used

    The bump here has slopes 1/60 beyond the standoff and 1/6 inside it.
    """

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        d = info["distance_to_target"]
        r = 2.0 * (self.prev_distance - d)
        if d >= cfg.standoff_distance:
            frac = 1.0 - (d - cfg.standoff_distance) / 60.0
        else:
            frac = 1.0 - (cfg.standoff_distance - d) / 6.0
        r += cfg.proximity_weight * max(0.0, frac)
        if info["docked"]:
            r += cfg.dock_bonus
        r += self._fuel_term(info, cfg.fuel_weight_mitigated)
        self.prev_distance = d
        return float(r)


class MitigatedReward(RewardFunction):
    """r = approach_shaping + dock_bonus_mitigated * safe_docked
          - velocity_penalty * velocity_violation - cone_penalty * cone_violation
          - fuel_weight_mitigated * fuel_used
    """

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        r = self._approach_shaping(info, cfg.shaping_weight)
        if info["phase"] == 2 and info["safe_docked"]:
            r += cfg.dock_bonus_mitigated

        if info["velocity_violation"]:
            r -= cfg.velocity_penalty
        if info["cone_violation"]:
            r -= cfg.cone_penalty
        r += self._fuel_term(info, cfg.fuel_weight_mitigated)

        self._update_prev(info)
        return float(r)


class PotentialBasedReward(RewardFunction):
    """r = F + dock_bonus_mitigated * safe_docked - velocity and cone penalties
          - fuel_weight_mitigated * fuel_used

    F = Phi(s') - Phi(s) with Phi = -potential_scale * (d + 8 v + 12 lateral).
    There is no discount factor in F. With the default config this reward equals
    MitigatedReward.
    """

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        r = self._approach_shaping(info, cfg.potential_scale)
        if info["safe_docked"]:
            r += cfg.dock_bonus_mitigated
        if info["velocity_violation"]:
            r -= cfg.velocity_penalty
        if info["cone_violation"]:
            r -= cfg.cone_penalty
        r += self._fuel_term(info, cfg.fuel_weight_mitigated)
        self._update_prev(info)
        return float(r)


class ConstraintGatedReward(RewardFunction):
    """r = gate * approach_shaping + dock_bonus_mitigated * safe_docked
          - fuel_weight_mitigated * fuel_used

    gate is gate_closed while the velocity corridor or the approach cone is
    violated, and 1 otherwise.
    """

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        gate = cfg.gate_closed if (info["velocity_violation"]
                                   or info["cone_violation"]) else 1.0

        r = gate * self._approach_shaping(info, cfg.shaping_weight)
        if info["safe_docked"]:
            r += cfg.dock_bonus_mitigated
        r += self._fuel_term(info, cfg.fuel_weight_mitigated)
        self._update_prev(info)
        return float(r)


class HierarchicalReward(RewardFunction):
    """The ``mitigated`` reward plus a one-time bonus of 10 on first reaching the
    close phase and 25 on first reaching the terminal phase."""

    PHASE_BONUS = {1: 10.0, 2: 25.0}

    def reset(self, initial_distance: float, initial_speed: float = 0.0,
              initial_lateral: float = 0.0) -> None:
        super().reset(initial_distance, initial_speed, initial_lateral)
        self.max_phase = 0

    def compute(self, state, action, info) -> float:
        cfg = self.cfg
        phase = info["phase"]
        r = 0.0

        if phase > self.max_phase:
            for p in range(self.max_phase + 1, phase + 1):
                r += self.PHASE_BONUS.get(p, 0.0)
            self.max_phase = phase

        r += self._approach_shaping(info, cfg.shaping_weight)
        if phase == 2 and info["safe_docked"]:
            r += cfg.dock_bonus_mitigated
        if info["velocity_violation"]:
            r -= cfg.velocity_penalty
        if info["cone_violation"]:
            r -= cfg.cone_penalty
        r += self._fuel_term(info, cfg.fuel_weight_mitigated)

        self._update_prev(info)
        return float(r)


REWARD_REGISTRY = {
    "hackable": HackableReward,
    "quota_hackable": QuotaHackableReward,
    "shaped_hackable": ShapedHackableReward,
    "mitigated": MitigatedReward,
    "potential": PotentialBasedReward,
    "constraint_gated": ConstraintGatedReward,
    "hierarchical": HierarchicalReward,
}

HACKABLE_TYPES = ("hackable", "quota_hackable", "shaped_hackable")
MITIGATED_TYPES = ("mitigated", "potential", "constraint_gated", "hierarchical")


def make_reward(reward_type: str, config: Optional[RewardConfig] = None,
                overrides: Optional[Dict[str, Any]] = None) -> RewardFunction:
    if reward_type not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward_type: {reward_type!r}. "
            f"Choose from {sorted(REWARD_REGISTRY)}."
        )
    cfg = (config or RewardConfig()).merged(overrides)
    return REWARD_REGISTRY[reward_type](cfg)
