"""
charts.py — Build the analytics figures for the train-until-perfect run.

All figures go to runs/charts/ as PNGs (150 dpi, labelled axes with
units). Data sources:
  - runs/perfect_run_log.jsonl         per-round eval summaries
  - runs/perfect_round_{n}_episodes.csv per-episode diagnostics

Figures:
  1. crash_ratio_vs_steps.png   failure + success rate vs training steps
  2. crash_causes_stacked.png   crash-cause counts per round (stacked)
  3. landing_positions.png      touchdown-x histogram + scatter vs start dist
  4. success_by_start.png       success rate by start-distance / start-speed bins
  5. eval_metrics_vs_steps.png   mean reward / episode length / fuel vs steps
  6. crash_anatomy.png          impact-speed + tilt distributions for failures
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostics import CRASH_CAUSES

CAUSE_COLORS = {
    "too-fast impact": "#e74c3c",
    "excessive tilt": "#f39c12",
    "off-pad landing": "#9b59b6",
    "out-of-bounds": "#3498db",
    "fuel exhaustion": "#1abc9c",
    "timeout": "#7f8c8d",
}
PAD_HALF = 0.18


def load_rounds(log_path="runs/perfect_run_log.jsonl"):
    rounds = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rounds.append(json.loads(line))
    return sorted(rounds, key=lambda r: r["round"])


def load_episodes(round_no: int):
    return pd.read_csv(f"runs/perfect_round_{round_no}_episodes.csv")


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def fig_crash_ratio(rounds, path):
    steps = [r["cum_steps"] for r in rounds]
    fail = [r["failure_rate"] * 100 for r in rounds]
    succ = [r["success_rate"] * 100 for r in rounds]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, succ, "o-", color="#2ecc71", label="Success (landed)")
    ax.plot(steps, fail, "o-", color="#e74c3c", label="Failure (any cause)")
    for x, y in zip(steps, succ):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    _style(ax, "Landing success vs training steps",
           "Cumulative training steps", "Episodes (%)")
    ax.set_ylim(-5, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_crash_causes(rounds, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"R{r['round']}" for r in rounds]
    bottom = np.zeros(len(rounds))
    for cause in CRASH_CAUSES:
        vals = np.array([r["crash_causes"].get(cause, 0) for r in rounds],
                        dtype=float)
        if vals.sum() == 0:
            continue
        ax.bar(labels, vals, bottom=bottom, label=cause,
               color=CAUSE_COLORS[cause])
        bottom += vals
    for i, r in enumerate(rounds):
        if r["failures"]:
            ax.text(i, r["failures"] + 15, str(r["failures"]),
                    ha="center", fontsize=9, weight="bold")
    _style(ax, "Crash-cause breakdown per eval round (1000 episodes each)",
           "Eval round", "Failed episodes")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_landing_positions(episodes: pd.DataFrame, path):
    landed = episodes[episodes["landed"]]
    failed_td = episodes[(~episodes["landed"])
                         & episodes["touchdown_x"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    bins = np.linspace(-1.0, 1.0, 41)
    ax.hist(landed["touchdown_x"], bins=bins, color="#2ecc71", alpha=0.85,
            label=f"Landed (n={len(landed)})")
    ax.hist(failed_td["touchdown_x"], bins=bins, color="#e74c3c", alpha=0.7,
            label=f"Crashed on ground (n={len(failed_td)})")
    ax.axvspan(-PAD_HALF, PAD_HALF, color="#2ecc71", alpha=0.12,
               label="Pad (±0.18)")
    _style(ax, "Touchdown x-position distribution",
           "Touchdown x (world units)", "Episodes")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.scatter(episodes["start_dist"], episodes["touchdown_x"],
               c=episodes["landed"].map({True: "#2ecc71", False: "#e74c3c"}),
               s=6, alpha=0.5)
    ax.axhspan(-PAD_HALF, PAD_HALF, color="#2ecc71", alpha=0.12)
    _style(ax, "Touchdown x vs starting distance from pad",
           "Starting distance from pad centre", "Touchdown x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _binned_success(episodes, key, bins, ax, title, xlabel):
    episodes = episodes.assign(bin=pd.cut(episodes[key], bins=bins))
    grp = episodes.groupby("bin", observed=True)
    rates = grp["landed"].mean() * 100
    counts = grp.size()
    centers = [iv.mid for iv in rates.index]
    widths = [iv.length * 0.8 for iv in rates.index]
    ax.bar(centers, rates, width=widths, color="#3498db",
           edgecolor="white")
    for c, r, n in zip(centers, rates, counts):
        ax.text(c, r + 2, f"{r:.0f}%\nn={n}", ha="center", fontsize=8)
    _style(ax, title, xlabel, "Success rate (%)")
    ax.set_ylim(0, 118)


def fig_success_by_start(episodes: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _binned_success(episodes, "start_dist",
                    [0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.3], axes[0],
                    "Success rate vs starting distance",
                    "Starting distance from pad centre")
    _binned_success(episodes, "start_speed",
                    [0, 0.15, 0.3, 0.45, 0.6, 0.9], axes[1],
                    "Success rate vs starting speed",
                    "Starting speed (m/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_eval_metrics(rounds, path):
    steps = [r["cum_steps"] for r in rounds]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].plot(steps, [r["mean_reward"] for r in rounds], "o-", color="#8e44ad")
    _style(axes[0], "Mean eval reward", "Training steps", "Reward")
    axes[1].plot(steps, [r["mean_episode_length"] for r in rounds], "o-",
                 color="#2980b9")
    _style(axes[1], "Mean episode length", "Training steps", "Steps")
    axes[2].plot(steps, [r["mean_fuel_used"] for r in rounds], "o-",
                 color="#27ae60")
    _style(axes[2], "Mean fuel used", "Training steps", "Fuel units (/100)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_crash_anatomy(episodes: pd.DataFrame, round_no: int, path):
    failed = episodes[~episodes["landed"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    ax = axes[0]
    ax.hist(failed["impact_vy"].abs(), bins=25, color="#e74c3c", alpha=0.85)
    ax.axvline(1.5, color="black", ls="--",
               label="Soft-landing limit (1.5 m/s)")
    _style(ax, f"Impact vertical speed — failures (round {round_no})",
           "|Vertical speed at impact| (m/s)", "Failed episodes")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(np.degrees(failed["impact_angle"].abs()), bins=25,
            color="#f39c12", alpha=0.85)
    ax.axvline(np.degrees(0.20), color="black", ls="--",
               label="Upright limit (~11.5°)")
    _style(ax, f"Impact tilt — failures (round {round_no})",
           "|Tilt at impact| (degrees)", "Failed episodes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_all_charts(rounds=None, outdir="runs/charts"):
    """Build every figure. Returns {name: path}."""
    os.makedirs(outdir, exist_ok=True)
    if rounds is None:
        rounds = load_rounds()
    final_round = max(r["round"] for r in rounds)
    final_eps = load_episodes(final_round)
    # Crash anatomy: last round that actually had failures.
    anatomy_round = final_round
    for r in sorted(rounds, key=lambda x: x["round"], reverse=True):
        if r["failures"] > 0:
            anatomy_round = r["round"]
            break
    anatomy_eps = (final_eps if anatomy_round == final_round
                   else load_episodes(anatomy_round))

    paths = {}
    jobs = [
        ("crash_ratio", lambda p: fig_crash_ratio(rounds, p)),
        ("crash_causes", lambda p: fig_crash_causes(rounds, p)),
        ("landing_positions", lambda p: fig_landing_positions(final_eps, p)),
        ("success_by_start", lambda p: fig_success_by_start(final_eps, p)),
        ("eval_metrics", lambda p: fig_eval_metrics(rounds, p)),
        ("crash_anatomy",
         lambda p: fig_crash_anatomy(anatomy_eps, anatomy_round, p)),
    ]
    for name, fn in jobs:
        path = os.path.join(outdir, name + ".png")
        fn(path)
        paths[name] = path
    return paths, final_round, anatomy_round


if __name__ == "__main__":
    paths, final_round, anatomy_round = build_all_charts()
    print(f"Built {len(paths)} charts (final round {final_round}, "
          f"anatomy round {anatomy_round}):")
    for name, path in paths.items():
        print(f"  {name}: {path}")
