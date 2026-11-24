"""Lane-keeping environment with the same reward families as the docking environment."""

from lane_keeping.env import LaneKeepingEnv
from lane_keeping.rewards import (LKREWARD_REGISTRY, make_lk_reward,
                                  LKRewardConfig)

__all__ = ["LaneKeepingEnv", "LKREWARD_REGISTRY", "make_lk_reward",
           "LKRewardConfig"]

import gymnasium

gymnasium.register(
    id="LaneKeeping-v0",
    entry_point="lane_keeping.env:LaneKeepingEnv",
)
