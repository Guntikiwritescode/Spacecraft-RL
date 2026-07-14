"""Cross-domain transfer of the reward-hacking taxonomy: autonomous driving.

A second, structurally different physical domain -- lane-keeping / route
completion with a kinematic bicycle model -- used to test whether the reward
misspecification failure modes (Cobra Effect, Soviet Quota Gaming) and their
mitigations are *domain-independent*.  The API mirrors
:mod:`goodharts_docking` so the same experiment harness drives both.
"""

from lane_keeping.env import LaneKeepingEnv
from lane_keeping.rewards import (LKREWARD_REGISTRY, make_lk_reward,
                                  LKRewardConfig)

__all__ = ["LaneKeepingEnv", "LKREWARD_REGISTRY", "make_lk_reward",
           "LKRewardConfig"]

import gymnasium

gymnasium.register(
    id="LaneKeeping-v0",
    entry_point="lane_keeping.env:LaneKeepingEnv",
    max_episode_steps=LaneKeepingEnv.MAX_STEPS,
)
