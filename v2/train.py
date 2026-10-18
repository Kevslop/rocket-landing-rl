"""
train.py — Train PPO on the 3D rocket landing env (v2).

Usage:
    .venv/bin/python v2/train.py [--timesteps 750000] [--n-envs 8] [--seed 0]

Saves the best model (by eval success rate) to v2/models/best3d.zip.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from env3d import RocketLanding3DEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
RUNS = os.path.join(HERE, "runs")


def make_env(seed: int):
    def _init():
        return RocketLanding3DEnv(seed=seed)

    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=750_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)

    vec = SubprocVecEnv(
        [make_env(args.seed + i) for i in range(args.n_envs)]
    )
    eval_env = RocketLanding3DEnv(seed=10_000)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODELS,
        log_path=RUNS,
        eval_freq=max(5_000 // args.n_envs, 500),
        n_eval_episodes=24,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        tensorboard_log=os.path.join(RUNS, "tb"),
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save(os.path.join(MODELS, "final3d"))
    # EvalCallback names the best model "best_model.zip"; rename for clarity.
    best_src = os.path.join(MODELS, "best_model.zip")
    best_dst = os.path.join(MODELS, "best3d.zip")
    if os.path.exists(best_src):
        os.replace(best_src, best_dst)
        print("best model ->", best_dst)


if __name__ == "__main__":
    main()
