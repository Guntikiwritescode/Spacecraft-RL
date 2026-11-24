"""Training callback that records docking outcomes next to the reward."""

from __future__ import annotations

from typing import List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    def __init__(self, wandb_run=None, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.docking_successes: List[bool] = []
        self.safe_docking_successes: List[bool] = []
        self._wandb = wandb_run

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for idx, done in enumerate(dones):
            if not done:
                continue
            info = infos[idx]
            docked = bool(info.get("docked", False))
            safe = bool(info.get("safe_docked", False))
            self.docking_successes.append(docked)
            self.safe_docking_successes.append(safe)
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(float(ep["r"]))
            if self._wandb is not None:
                self._wandb.log({
                    "train/docked": int(docked),
                    "train/safe_docked": int(safe),
                    "train/episode_reward": float(ep["r"]) if ep else np.nan,
                    "train/episode_length": int(ep["l"]) if ep else 0,
                    "global_step": int(self.num_timesteps),
                })
        return True

    def summary(self) -> dict:
        n = len(self.docking_successes)
        return {
            "total_episodes": n,
            "docking_success_rate": (sum(self.docking_successes) / n) if n else 0.0,
            "safe_docking_rate": (sum(self.safe_docking_successes) / n) if n else 0.0,
            "final_mean_reward": float(np.mean(self.episode_rewards[-20:]))
            if self.episode_rewards else float("nan"),
        }
