"""
train_mission.py — Train one loadout-conditioned PPO policy for the full
takeoff -> waypoint -> landing mission (v2).

Each training env resamples its engine/tech loadout across episodes, so the
single policy learns to fly all 16 loadouts (it observes the loadout's key
parameters). Weather is domain-randomized per episode.

Curriculum:
    Stage 1: --spawn air    (start airborne near the waypoint: learn to
                             steer, slow down, and capture)
    Stage 2: --spawn ground --warmstart models/stage1.zip
                             (full takeoff -> waypoint -> landing mission)

Usage:
    .venv/bin/python v2/train_mission.py --spawn air --timesteps 600000 \\
        --run-name stage1 --seed 0
    .venv/bin/python v2/train_mission.py --spawn ground \\
        --warmstart v2/models/stage1.zip --timesteps 1500000 \\
        --run-name best_mission --seed 1

Best model -> v2/models/<run-name>.zip
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from mission_env import LoadoutMissionEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
RUNS = os.path.join(HERE, "runs")


def make_env(seed: int, spawn: str, pad_r: float):
    def _init():
        return LoadoutMissionEnv(
            engine_id="raptor3", tech_id="standard",
            weather="auto", seed=seed, randomize_loadout=True,
            spawn=spawn, pad_r=pad_r,
        )

    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_200_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spawn", default="ground",
                    choices=["ground", "air", "landing"])
    ap.add_argument("--pad-r", type=float, default=7.0,
                    help="landing pad radius (m); curriculum: start large")
    ap.add_argument("--warmstart", default=None)
    ap.add_argument("--run-name", default="best_mission")
    args = ap.parse_args()

    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)

    vec = SubprocVecEnv(
        [make_env(args.seed + i, args.spawn, args.pad_r)
         for i in range(args.n_envs)])
    eval_env = LoadoutMissionEnv(
        engine_id="merlin1d", tech_id="standard",
        weather="auto", seed=99_000, randomize_loadout=True,
        spawn=args.spawn, pad_r=args.pad_r,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODELS,
        log_path=RUNS,
        eval_freq=max(10_000 // args.n_envs, 500),
        n_eval_episodes=20,
        deterministic=True,
    )

    if args.warmstart:
        print("warm-starting from", args.warmstart, flush=True)
        model = PPO.load(args.warmstart, env=vec)
    else:
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
        )
    model.learn(total_timesteps=args.timesteps, callback=eval_cb,
                reset_num_timesteps=bool(args.warmstart))
    model.save(os.path.join(MODELS, args.run_name + "_final"))
    best_src = os.path.join(MODELS, "best_model.zip")
    best_dst = os.path.join(MODELS, args.run_name + ".zip")
    if os.path.exists(best_src):
        os.replace(best_src, best_dst)
        print("best model ->", best_dst, flush=True)


if __name__ == "__main__":
    main()
