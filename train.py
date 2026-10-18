"""
train.py — Train a PPO agent on the rocket landing environment.

Usage:
    python train.py                    # default: 200k timesteps
    python train.py --timesteps 500000
    python train.py --smoke               # quick smoke test (~4k timesteps)
    python train.py --wind 1.5 --out models/windy
    python train.py --load models/ppo_rocket.zip --timesteps 100000

Outputs (under models/ by default):
    <name>.zip              trained PPO policy
    <name>_curve.csv        per-episode reward / length / landing stats
    <name>_config.json      hyperparameters used

Everything runs locally with stable-baselines3 + PyTorch (CPU is fine).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from env import RocketLandingEnv


def make_env(wind: float, seed: int):
    """Build one env instance. Kept as a function so SB3's Monitor/DummyVec
    wrappers can call it, and so evaluation can reuse the same setup."""
    return RocketLandingEnv(wind=wind, seed=seed)


def train(timesteps: int = 200_000, wind: float = 0.0, out: str = "models/ppo_rocket",
          load: str | None = None, seed: int = 0, verbose: bool = True):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    class CurveLogger(BaseCallback):
        """Logs per-episode reward, length, landed/crashed to a CSV file."""

        def __init__(self, csv_path: str):
            super().__init__()
            self.csv_path = csv_path
            self._ep_reward = 0.0
            self._ep_len = 0
            self._file = open(csv_path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(["episode", "reward", "length", "landed", "crashed"])

        def _on_step(self) -> bool:
            self._ep_reward += float(self.locals["rewards"][0])
            self._ep_len += 1
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    self._writer.writerow([
                        self.n_calls, round(self._ep_reward, 2), self._ep_len,
                        int(info.get("landed", False)),
                        int(info.get("crashed", False)),
                    ])
                    self._ep_reward = 0.0
                    self._ep_len = 0
            return True

        def _on_training_end(self) -> None:
            self._file.close()

    env = Monitor(make_env(wind=wind, seed=seed))
    curve_path = out + "_curve.csv"
    callback = CurveLogger(curve_path)

    config = {
        "algorithm": "PPO",
        "timesteps": timesteps,
        "wind": wind,
        "seed": seed,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy": "MlpPolicy",
        "policy_kwargs": {"net_arch": [128, 128]},
    }

    if load:
        print(f"Continuing training from {load}")
        model = PPO.load(load, env=env)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            policy_kwargs=config["policy_kwargs"],
            seed=seed, verbose=0,
        )

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    elapsed = time.time() - t0

    model.save(out)
    with open(out + "_config.json", "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"Saved model to {out}.zip ({elapsed:.1f}s, "
              f"{timesteps / max(elapsed, 1e-9):.0f} steps/s)")
        print(f"Training curve: {curve_path}")
    return out + ".zip"


def main():
    parser = argparse.ArgumentParser(description="Train PPO on rocket landing.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--wind", type=float, default=0.0,
                        help="constant horizontal wind acceleration (m/s^2)")
    parser.add_argument("--out", type=str, default="models/ppo_rocket")
    parser.add_argument("--load", type=str, default=None,
                        help="continue training from an existing .zip model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="quick smoke test: 4k timesteps")
    args = parser.parse_args()

    timesteps = 4_000 if args.smoke else args.timesteps
    out = args.out + ("_smoke" if args.smoke else "")
    train(timesteps=timesteps, wind=args.wind, out=out, load=args.load,
          seed=args.seed)


if __name__ == "__main__":
    main()
