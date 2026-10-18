"""
diagnostics.py — Deterministic per-episode evaluation with rich logging.

Runs a trained model for N episodes with fixed seeds (seed_base + i),
collecting the full per-episode diagnostics the environment records
(starting conditions, impact velocity/tilt, touchdown position, fuel,
crash cause). Used by train_until_perfect.py, charts.py, and the
--diag-csv option of evaluate.py.

The eval seed base must differ from anything used in training so the
evaluation measures generalisation, not memorised starts.
"""

from __future__ import annotations

from env import RocketLandingEnv

CRASH_CAUSES = [
    "too-fast impact",
    "excessive tilt",
    "off-pad landing",
    "out-of-bounds",
    "fuel exhaustion",
    "timeout",
]


def run_eval_episodes(model, episodes: int, seed_base: int, wind: float = 0.0):
    """Run `episodes` deterministic episodes; return a list of diag dicts."""
    env = RocketLandingEnv(wind=wind)
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            env.last_action = action
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        diag = dict(info.get("diag", {}))
        diag["episode"] = i
        diag["seed"] = seed_base + i
        diag["reward"] = float(ep_reward)
        rows.append(diag)
    return rows


def summarize(rows):
    """Aggregate a list of per-episode diag dicts into headline stats."""
    n = len(rows)
    landed = sum(1 for r in rows if r.get("landed"))
    failures = n - landed
    cause_counts = {c: 0 for c in CRASH_CAUSES}
    for r in rows:
        if not r.get("landed"):
            cause_counts[r.get("crash_cause", "timeout")] += 1
    import statistics

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(statistics.fmean(vals)) if vals else 0.0

    return {
        "episodes": n,
        "landed": landed,
        "failures": failures,
        "success_rate": landed / n if n else 0.0,
        "failure_rate": failures / n if n else 0.0,
        "crash_causes": cause_counts,
        "mean_reward": mean("reward"),
        "mean_fuel_used": mean("fuel_used"),
        "mean_episode_length": mean("steps"),
        "mean_start_dist": mean("start_dist"),
        "mean_start_speed": mean("start_speed"),
    }
