"""Lane-keeping rewards with the same family names as the docking rewards."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class LKRewardConfig:
    centering_weight: float = 1.0
    center_scale: float = 0.8         # [m]
    progress_weight: float = 0.3
    progress_weight_mitigated: float = 2.0
    odometer_weight: float = 1.0
    completion_bonus: float = 5.0
    completion_bonus_mitigated: float = 100.0
    time_penalty: float = 0.05
    lane_penalty: float = 2.0
    speed_penalty: float = 1.0

    def merged(self, overrides: Optional[Dict[str, Any]]) -> "LKRewardConfig":
        if not overrides:
            return self
        known = {k: v for k, v in overrides.items() if k in self.__dataclass_fields__}
        return replace(self, **known)


class LKRewardFunction:
    def __init__(self, config: Optional[LKRewardConfig] = None):
        self.cfg = config or LKRewardConfig()
        self.prev_s = 0.0

    def reset(self, initial_progress: float = 0.0):
        self.prev_s = float(initial_progress)

    def compute(self, info) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


class LKHackableReward(LKRewardFunction):
    """r = centering_weight * exp(-0.5 (y / center_scale)^2)
          + progress_weight * (progress made) - lane_penalty * off_lane
          + completion_bonus * completed
    """

    def compute(self, info) -> float:
        cfg = self.cfg
        y = info["lateral_offset"]
        centering = cfg.centering_weight * float(np.exp(-0.5 * (y / cfg.center_scale) ** 2))
        progress = cfg.progress_weight * (info["progress"] - self.prev_s)
        r = centering + progress
        if info["off_lane"]:
            r -= cfg.lane_penalty
        if info["completed"]:
            r += cfg.completion_bonus
        self.prev_s = info["progress"]
        return float(r)


class LKQuotaReward(LKRewardFunction):
    """r = odometer_weight * (path length this step) + completion_bonus * completed"""

    def compute(self, info) -> float:
        cfg = self.cfg
        r = cfg.odometer_weight * info["distance_travelled_this_step"]
        if info["completed"]:
            r += cfg.completion_bonus
        self.prev_s = info["progress"]
        return float(r)


class LKMitigatedReward(LKRewardFunction):
    """r = progress_weight_mitigated * (progress made) - time_penalty
          - lane_penalty * off_lane - speed_penalty * over_speed
          + completion_bonus_mitigated * safe_completed
    """

    def compute(self, info) -> float:
        cfg = self.cfg
        r = cfg.progress_weight_mitigated * (info["progress"] - self.prev_s)
        r -= cfg.time_penalty
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
