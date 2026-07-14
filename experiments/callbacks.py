"""Training callbacks that track *mission outcomes* separately from reward.

The whole point of the study is that reward and mission success can diverge.
SB3's default logging reports mean episode reward; :class:`MetricsCallback`
additionally records whether each finished episode actually docked (ground
truth from the ``info`` dict), so we can watch reward climb while docking stays
at zero -- the signature of reward hacking.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """Records per-episode docking success, safe-docking, reward and length.

    Works with vectorised or single environments (reads ``dones``/``infos`` from
    ``self.locals``).  Optionally mirrors metrics to Weights & Biases.
    """

    def __init__(self, wandb_run=None, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.docking_successes: List[bool] = []
        self.safe_docking_successes: List[bool] = []
        self.timesteps_at_episode: List[int] = []
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
            self.timesteps_at_episode.append(int(self.num_timesteps))
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(float(ep["r"]))
                self.episode_lengths.append(int(ep["l"]))
            if self._wandb is not None:
                self._wandb.log({
                    "train/docked": int(docked),
                    "train/safe_docked": int(safe),
                    "train/episode_reward": float(ep["r"]) if ep else np.nan,
                    "train/episode_length": int(ep["l"]) if ep else 0,
                    "global_step": int(self.num_timesteps),
                })
        return True

    # --- convenience accessors used by the plotting code -------------------- #
    def rolling_success(self, window: int = 20) -> Optional[np.ndarray]:
        s = np.asarray(self.docking_successes, dtype=float)
        if len(s) <= window:
            return None
        return np.convolve(s, np.ones(window) / window, mode="valid")

    def summary(self) -> dict:
        n = len(self.docking_successes)
        return {
            "total_episodes": n,
            "docking_success_rate": (sum(self.docking_successes) / n) if n else 0.0,
            "safe_docking_rate": (sum(self.safe_docking_successes) / n) if n else 0.0,
            "final_mean_reward": float(np.mean(self.episode_rewards[-20:]))
            if self.episode_rewards else float("nan"),
        }
