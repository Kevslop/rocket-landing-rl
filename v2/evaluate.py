"""
evaluate.py — Evaluate a trained v2 policy over N episodes.

Usage:
    .venv/bin/python v2/evaluate.py [--model v2/models/best3d.zip] [--episodes 1000]

Writes results to v2/eval_results.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

from stable_baselines3 import PPO

from env3d import RocketLanding3DEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "best3d.zip"))
    ap.add_argument("--episodes", type=int, default=1000)
    args = ap.parse_args()

    model = PPO.load(args.model)
    outcomes: dict[str, int] = {}
    touchdown_vz, touchdown_vxy, fuel_used = [], [], []
    t0 = time.time()
    for ep in range(args.episodes):
        env = RocketLanding3DEnv(seed=50_000 + ep)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        oc = info.get("outcome") or "unknown"
        outcomes[oc] = outcomes.get(oc, 0) + 1
        if oc == "landed":
            touchdown_vz.append(info.get("vz", 0.0))
            touchdown_vxy.append(info.get("vxy", 0.0))
        fuel_used.append(1.0 - info.get("fuel_frac", 0.0))

    n = args.episodes
    landed = outcomes.get("landed", 0)
    results = {
        "model": args.model,
        "episodes": n,
        "success_rate": landed / n,
        "outcomes": outcomes,
        "mean_touchdown_vz": float(np.mean(touchdown_vz)) if touchdown_vz else None,
        "mean_touchdown_vxy": float(np.mean(touchdown_vxy)) if touchdown_vxy else None,
        "mean_fuel_used_frac": float(np.mean(fuel_used)),
        "eval_seconds": time.time() - t0,
    }
    out = os.path.join(HERE, "eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
