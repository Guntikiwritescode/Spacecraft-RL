"""Goodhart's Taxonomy: reward misspecification in safety-critical RL.

A parameterisable 3D spacecraft docking environment (full Clohessy-Wiltshire
relative orbital dynamics) for studying reward hacking, plus a taxonomy of
reward architectures spanning exploitable and mitigated specifications.
"""

from goodharts_docking.dynamics import CWDynamics, mean_motion
from goodharts_docking.env import DockingEnv
from goodharts_docking.rewards import (
    REWARD_REGISTRY,
    HACKABLE_TYPES,
    MITIGATED_TYPES,
    RewardConfig,
    HackableReward,
    QuotaHackableReward,
    MitigatedReward,
    PotentialBasedReward,
    ConstraintGatedReward,
    HierarchicalReward,
    make_reward,
)

__version__ = "0.1.0"

__all__ = [
    "DockingEnv",
    "CWDynamics",
    "mean_motion",
    "RewardConfig",
    "REWARD_REGISTRY",
    "HACKABLE_TYPES",
    "MITIGATED_TYPES",
    "HackableReward",
    "QuotaHackableReward",
    "MitigatedReward",
    "PotentialBasedReward",
    "ConstraintGatedReward",
    "HierarchicalReward",
    "make_reward",
]

import gymnasium

# Register every reward variant as its own Gymnasium id for convenience.
gymnasium.register(
    id="GoodhartsDocking-v0",
    entry_point="goodharts_docking.env:DockingEnv",
    max_episode_steps=DockingEnv.MAX_STEPS,
)

for _rt in REWARD_REGISTRY:
    gymnasium.register(
        id=f"GoodhartsDocking-{_rt}-v0",
        entry_point="goodharts_docking.env:DockingEnv",
        max_episode_steps=DockingEnv.MAX_STEPS,
        kwargs={"reward_type": _rt},
    )
