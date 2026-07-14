"""Train PPO against the hackable reward and watch the Cobra Effect emerge.

The dense reward looks reasonable, but the non-telescoping proximity term plus
episode-termination-on-success makes *hovering near the port* more valuable than
docking.  The agent's reward climbs while its docking-success rate stays at zero
-- reward hacking, quantified.

    python -m experiments.train_cobra_effect [--timesteps N] [--seed S]
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from goodharts_docking import DockingEnv
from experiments.callbacks import MetricsCallback
from experiments.common import MODELS_DIR, CAMPAIGN_DISTANCE
import os


def train(timesteps: int = 100_000, seed: int = 42):
    env = DockingEnv(reward_type="hackable", initial_distance=CAMPAIGN_DISTANCE)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, device="cpu",
                learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                policy_kwargs=dict(net_arch=[64, 64]))
    callback = MetricsCallback()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

    path = os.path.join(MODELS_DIR, "ppo_cobra_effect")
    model.save(path)

    s = callback.summary()
    print("\n=== COBRA EFFECT TRAINING RESULTS ===")
    print(f"Total episodes:        {s['total_episodes']}")
    print(f"Docking success rate:  {s['docking_success_rate']:.1%}")
    print(f"Final mean reward:     {s['final_mean_reward']:.1f}")
    print("(Rising reward with ~0% docking = reward hacking confirmed)")
    print(f"Model saved -> {path}.zip")
    return model, callback


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    train(**vars(p.parse_args()))
