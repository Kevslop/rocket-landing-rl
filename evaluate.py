"""
evaluate.py — Evaluate a trained PPO model on the rocket landing task.

Usage:
    python evaluate.py                          # 50 episodes, models/ppo_rocket.zip
    python evaluate.py --model models/ppo_rocket_smoke.zip --episodes 20
    python evaluate.py --episodes 100 --gif runs/eval_landing.gif --seed 123

Outputs:
    - Prints success rate (landed on pad), crash rate, mean reward,
      mean fuel used, mean episode length.
    - Saves a GIF of the first landed episode (or the first episode if none
      landed) to --gif (default: runs/eval_landing.gif). GIF frames are
      captured every 2nd step to keep the file small.
    - Appends a one-line JSON summary to runs/eval_summary.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from env import RocketLandingEnv
from render import frame_array, save_gif


def evaluate(model_path: str, episodes: int = 50, seed: int = 0,
             gif_path: str = "runs/eval_landing.gif", wind: float = 0.0,
             record_every: int = 2):
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = RocketLandingEnv(wind=wind)

    landed = crashed = 0
    rewards, fuels, lengths = [], [], []
    gif_frames: list[np.ndarray] = []
    gif_saved = False

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_frames: list[np.ndarray] = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Stash the action so the renderer can draw the flame.
            env.last_action = action
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            if len(ep_frames) < 300 and (env.steps % record_every == 0):
                ep_frames.append(frame_array(env))
        rewards.append(ep_reward)
        fuels.append(RocketLandingEnv.FUEL_MAX - info["fuel_left"])
        lengths.append(info["steps"])
        if info["landed"]:
            landed += 1
        if info["crashed"]:
            crashed += 1
        # Keep frames from the first landed episode; fall back to episode 0.
        if not gif_saved and (info["landed"] or ep == 0):
            gif_frames = ep_frames
            gif_saved = True

    stats = {
        "model": model_path,
        "episodes": episodes,
        "success_rate": landed / episodes,
        "crash_rate": crashed / episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_fuel_used": float(np.mean(fuels)),
        "mean_episode_length": float(np.mean(lengths)),
        "landed": landed,
        "crashed": crashed,
        "wind": wind,
    }

    if gif_frames:
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        save_gif(gif_frames, gif_path)
        stats["gif"] = gif_path

    # Append a JSON-lines summary for tracking runs over time.
    os.makedirs("runs", exist_ok=True)
    with open("runs/eval_summary.jsonl", "a") as f:
        f.write(json.dumps(stats) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained rocket agent.")
    parser.add_argument("--model", type=str, default="models/ppo_rocket.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wind", type=float, default=0.0)
    parser.add_argument("--gif", type=str, default="runs/eval_landing.gif")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--diag-csv", type=str, default=None,
                        help="write per-episode diagnostics to this CSV")
    args = parser.parse_args()

    gif_path = args.gif if not args.no_gif else ""
    stats = evaluate(args.model, episodes=args.episodes, seed=args.seed,
                     gif_path=gif_path or "runs/eval_landing.gif",
                     wind=args.wind)
    if args.diag_csv:
        from diagnostics import run_eval_episodes
        from stable_baselines3 import PPO as _PPO
        _model = _PPO.load(args.model)
        rows = run_eval_episodes(_model, args.episodes,
                                 seed_base=1_000_000, wind=args.wind)
        import csv as _csv
        fields = ["episode", "seed", "reward", "start_x", "start_y",
                  "start_vx", "start_vy", "start_angle", "start_ang_vel",
                  "start_speed", "start_dist", "impact_vx", "impact_vy",
                  "impact_angle", "touchdown_x", "final_x", "final_y",
                  "fuel_used", "fuel_left", "steps", "landed", "crashed",
                  "truncated", "crash_cause", "outcome"]
        with open(args.diag_csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})
        print(f"  Diagnostics CSV:       {args.diag_csv}")
    print(f"Model: {stats['model']}  ({stats['episodes']} episodes)")
    print(f"  Success (landed on pad): {stats['landed']}/{stats['episodes']} "
          f"= {stats['success_rate']:.1%}")
    print(f"  Crashes:               {stats['crashed']}/{stats['episodes']} "
          f"= {stats['crash_rate']:.1%}")
    print(f"  Mean reward:           {stats['mean_reward']:.1f}")
    print(f"  Mean fuel used:        {stats['mean_fuel_used']:.1f} / "
          f"{RocketLandingEnv.FUEL_MAX:.0f}")
    print(f"  Mean episode length:   {stats['mean_episode_length']:.0f} steps")
    if stats.get("gif"):
        print(f"  Landing GIF:           {stats['gif']}")


if __name__ == "__main__":
    main()
