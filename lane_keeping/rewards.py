"""Reward taxonomy for lane-keeping -- the cross-domain mirror of the docking
taxonomy.  The same two failure modes recur:

* :class:`LKHackableReward`  -- the *Cobra Effect*.  A per-step lane-centering
  "hold" bonus, plus episode-termination on route completion, makes *dawdling
  centred in the lane* more valuable than finishing the route.  The car crawls
  or stops rather than completing the trip -- it farms centred-time.
* :class:`LKQuotaReward`     -- *Soviet Quota Gaming*.  Rewarding distance
  travelled (the odometer) makes the car maximise mileage by speeding, so it
  leaves the lane on curves -- reckless "miles-driven" optimisation.
* :class:`LKMitigatedReward` -- sparse, safe-completion-dominant reward with a
  time penalty and lane/speed constraint penalties.  The car completes the
  route safely.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class LKRewardConfig:
    centering_weight: float = 1.0     # Cobra-Effect farmable surface
    center_scale: float = 0.8         # [m] width of the centring bonus
    progress_weight: float = 0.3      # telescoping progress guidance
    odometer_weight: float = 1.0      # quota-gaming farmable surface
    completion_bonus: float = 5.0     # hackable: small terminal bonus
    completion_bonus_mitigated: float = 100.0
    time_penalty: float = 0.2
    lane_penalty: float = 2.0
    speed_penalty: float = 1.0

    def merged(self, overrides: Optional[Dict[str, Any]]) -> "LKRewardConfig":
        if not overrides:
            return self
        known = {k: v for k, v in overrides.items() if k in self.__dataclass_fields__}
        return replace(self, **known)


class LKRewardFunction:
    family = "unknown"

    def __init__(self, config: Optional[LKRewardConfig] = None):
        self.cfg = config or LKRewardConfig()
        self.prev_s = 0.0

    def reset(self, initial_progress: float = 0.0):
        self.prev_s = float(initial_progress)

    def compute(self, info) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


class LKHackableReward(LKRewardFunction):
    """Cobra Effect: farm centred-time instead of completing the route."""

    family = "hackable"

    def compute(self, info) -> float:
        cfg = self.cfg
        y = info["lateral_offset"]
        centering = cfg.centering_weight * float(np.exp(-0.5 * (y / cfg.center_scale) ** 2))
        progress = cfg.progress_weight * (info["progress"] - self.prev_s)
        r = centering + progress
        if info["off_lane"]:
            r -= cfg.lane_penalty
        if info["completed"]:
            r += cfg.completion_bonus       # small -> completing is not worth it
        self.prev_s = info["progress"]
        return float(r)


class LKQuotaReward(LKRewardFunction):
    """Soviet Quota Gaming: maximise the odometer, ignore lane safety."""

    family = "hackable"

    def compute(self, info) -> float:
        cfg = self.cfg
        r = cfg.odometer_weight * info["distance_travelled_this_step"]
        if info["completed"]:
            r += cfg.completion_bonus
        self.prev_s = info["progress"]
        return float(r)


class LKMitigatedReward(LKRewardFunction):
    """Safe-completion-dominant reward: finish the route, in-lane, sensibly."""

    family = "mitigated"

    def compute(self, info) -> float:
        cfg = self.cfg
        r = cfg.progress_weight * (info["progress"] - self.prev_s)   # telescoping
        r -= cfg.time_penalty                                         # time costs
        if info["off_lane"]:
            r -= cfg.lane_penalty
        if info["over_speed"]:
            r -= cfg.speed_penalty
        if info["safe_completed"]:
            r += cfg.completion_bonus_mitigated
        self.prev_s = info["progress"]
        return float(r)


LKREWARD_REGISTRY = {
    "hackable": LKHackableReward,
    "quota_hackable": LKQuotaReward,
    "mitigated": LKMitigatedReward,
}


def make_lk_reward(reward_type: str, config: Optional[LKRewardConfig] = None,
                   overrides: Optional[Dict[str, Any]] = None) -> LKRewardFunction:
    if reward_type not in LKREWARD_REGISTRY:
        raise ValueError(f"Unknown lane-keeping reward {reward_type!r}; "
                         f"choose from {sorted(LKREWARD_REGISTRY)}")
    cfg = (config or LKRewardConfig()).merged(overrides)
    return LKREWARD_REGISTRY[reward_type](cfg)
