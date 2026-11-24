"""Spacecraft docking environment and reward functions."""

from spacecraft_rl.dynamics import CWDynamics, mean_motion
from spacecraft_rl.env import DockingEnv
from spacecraft_rl.rewards import (
    REWARD_REGISTRY,
    HACKABLE_TYPES,
    MITIGATED_TYPES,
    RewardConfig,
    HackableReward,
    QuotaHackableReward,
    ShapedHackableReward,
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
    "ShapedHackableReward",
    "MitigatedReward",
    "PotentialBasedReward",
    "ConstraintGatedReward",
    "HierarchicalReward",
    "make_reward",
]

import gymnasium

# The environment truncates episodes itself at ``max_steps``, so no TimeLimit
# wrapper is registered. A wrapper would ignore a ``max_steps`` kwarg.
gymnasium.register(
    id="SpacecraftDocking-v0",
    entry_point="spacecraft_rl.env:DockingEnv",
)

for _rt in REWARD_REGISTRY:
    gymnasium.register(
        id=f"SpacecraftDocking-{_rt}-v0",
        entry_point="spacecraft_rl.env:DockingEnv",
        kwargs={"reward_type": _rt},
    )
