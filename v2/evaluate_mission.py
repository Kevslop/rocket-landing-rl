"""
evaluate_mission.py — Evaluate a pilot on every loadout:
"can we take off and land efficiently with this loadout?"

Pilot: --pilot scripted (default; PD controller) or --pilot ppo.

Usage:
    .venv/bin/python v2/evaluate_mission.py [--episodes 60] [--pilot scripted]

Writes v2/eval_mission.json with per-loadout success rate, fuel used,
flight time, waypoint rate, and outcome breakdown.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from engines import list_loadouts
from mission_env import LoadoutMissionEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--pilot", default="scripted", choices=["scripted", "ppo"])
    ap.add_argument("--model", default=os.path.join(HERE, "models",
                                                    "best_mission.zip"))
    ap.add_argument("--out", default=os.path.join(HERE, "eval_mission.json"))
    args = ap.parse_args()

    if args.pilot == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        print("pilot: PPO", args.model, flush=True)
    else:
        from scripted import scripted_action
        model = None
        print("pilot: scripted PD controller", flush=True)

    results = {"pilot": args.pilot, "model": args.model,
               "episodes_per_loadout": args.episodes, "loadouts": {}}
    t0 = time.time()
    for eng, tech in list_loadouts():
        key = f"{eng}+{tech}"
        outcomes, fuels, times, wps = {}, [], [], 0
        for ep in range(args.episodes):
            env = LoadoutMissionEnv(engine_id=eng, tech_id=tech,
                                    weather="auto", seed=77_000 + ep)
            obs, _ = env.reset()
            done = False
            while not done:
                if args.pilot == "ppo":
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = scripted_action(env)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            oc = info.get("outcome") or "unknown"
            outcomes[oc] = outcomes.get(oc, 0) + 1
            fuels.append(info["fuel_used_frac"])
            times.append(info["flight_time"])
            wps += 1 if info["waypoint_reached"] else 0
        n = args.episodes
        results["loadouts"][key] = {
            "engine": eng, "tech": tech,
            "success_rate": round(outcomes.get("landed", 0) / n, 3),
            "waypoint_rate": round(wps / n, 3),
            "mean_fuel_used_pct": round(float(np.mean(fuels)) * 100, 1),
            "mean_flight_time_s": round(float(np.mean(times)), 1),
            "outcomes": outcomes,
        }
        print(f"{key:28s} landed {outcomes.get('landed',0):3d}/{n} "
              f"fuel {np.mean(fuels)*100:5.1f}% t {np.mean(times):5.1f}s",
              flush=True)

    results["eval_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
