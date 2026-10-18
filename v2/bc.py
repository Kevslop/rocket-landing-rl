"""
bc.py — Behavioral cloning from the scripted controller, then PPO fine-tune.

1. Generate demonstrations with scripted.py's controller.
2. Supervised-train the PPO actor to mimic (MSE loss).
3. Save as a warmstart for PPO fine-tuning.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch as th

from stable_baselines3 import PPO

from mission_env import LoadoutMissionEnv
from scripted import scripted_action

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")


def collect_demos(n_episodes=200, spawn="landing", pad_r=7.0,
                  success_only=False):
    obs_list, act_list = [], []
    ep = 0
    collected = 0
    while collected < n_episodes:
        env = LoadoutMissionEnv(
            engine_id="raptor3", tech_id="standard", weather="auto",
            seed=60_000 + ep, randomize_loadout=True,
            spawn=spawn, pad_r=pad_r)
        obs, _ = env.reset()
        done = False
        ep_obs, ep_act = [], []
        while not done:
            act = scripted_action(env)
            ep_obs.append(obs.copy())
            ep_act.append(act.copy())
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        ep += 1
        if success_only and info["outcome"] != "landed":
            continue
        obs_list.extend(ep_obs)
        act_list.extend(ep_act)
        collected += 1
        if collected % 20 == 0:
            print(f"  {collected}/{n_episodes} demos (tried {ep} eps)",
                  flush=True)
    return np.array(obs_list, dtype=np.float32), np.array(act_list,
                                                         dtype=np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--spawn", default="landing",
                    choices=["ground", "air", "landing"])
    ap.add_argument("--pad-r", type=float, default=7.0)
    ap.add_argument("--success-only", action="store_true",
                    help="only keep demonstrations that landed")
    ap.add_argument("--out", default="bc_landing.zip")
    args = ap.parse_args()

    print("collecting demonstrations...", flush=True)
    obs, act = collect_demos(args.demos, args.spawn, args.pad_r,
                             args.success_only)
    print(f"{len(obs)} samples", flush=True)

    # fresh PPO (actor only will be trained)
    env = LoadoutMissionEnv(spawn=args.spawn, pad_r=args.pad_r)
    model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]),
                verbose=0)

    # supervised training on the actor
    opt = th.optim.Adam(model.policy.parameters(), lr=1e-3)
    obs_t = th.from_numpy(obs)
    act_t = th.from_numpy(act)
    ds = th.utils.data.TensorDataset(obs_t, act_t)
    loader = th.utils.data.DataLoader(ds, batch_size=512, shuffle=True)

    model.policy.train()
    for epoch in range(args.epochs):
        total = 0.0
        for b_obs, b_act in loader:
            # forward through the policy's action head
            features = model.policy.extract_features(b_obs)
            latent_pi, _ = model.policy.mlp_extractor(features)
            pred = model.policy.action_net(latent_pi)
            # PPO squashes with tanh; our demos are in [-1,1] already
            pred_a = th.tanh(pred)
            loss = th.nn.functional.mse_loss(pred_a, b_act)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b_obs)
        if epoch % 10 == 0:
            print(f"epoch {epoch}: mse={total/len(obs):.5f}", flush=True)

    out = os.path.join(MODELS, args.out)
    model.save(out)
    print("saved", out, flush=True)

    # quick eval
    from collections import Counter
    c = Counter()
    for ep in range(20):
        env = LoadoutMissionEnv(engine_id="raptor3", tech_id="standard",
                                weather="clear", seed=70_000 + ep,
                                spawn=args.spawn, pad_r=args.pad_r)
        obs, _ = env.reset()
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        c[info["outcome"]] += 1
    print("bc eval:", dict(c))


if __name__ == "__main__":
    main()
