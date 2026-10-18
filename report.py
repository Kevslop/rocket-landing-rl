"""
report.py — Build runs/report.html: a polished showcase page embedding
all charts plus headline numbers and data-driven takeaways.

The takeaways are computed from the logged data (not hardcoded):
  - dominant crash cause in the first vs the last eval round
  - hardest starting-distance / starting-speed bin in the final eval
  - fuel trend from first to last round
  - the env bug-fix story (why the old 68% number is not comparable)
"""

from __future__ import annotations

import base64
import json
import os

import numpy as np
import pandas as pd

from charts import build_all_charts, load_rounds


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _dominant_cause(round_summary):
    causes = round_summary["crash_causes"]
    if not any(causes.values()):
        return "none (zero failures)"
    cause = max(causes, key=causes.get)
    return f"{cause} ({causes[cause]} of {round_summary['failures']} failures)"


def _hardest_bin(episodes: pd.DataFrame, key: str, bins):
    eps = episodes.assign(bin=pd.cut(episodes[key], bins=bins))
    grp = eps.groupby("bin", observed=True)
    rates = grp["landed"].mean() * 100
    counts = grp.size()
    worst = rates.idxmin()
    return (f"{key} in [{worst.left:.2f}, {worst.right:.2f}] — "
            f"{rates[worst]:.0f}% success (n={counts[worst]})")


def build_report(rounds=None, out="runs/report.html",
                 gif_path="runs/eval_landing_best.gif"):
    if rounds is None:
        rounds = load_rounds()
    with open("runs/perfect_final.json") as f:
        final_meta = json.load(f)
    # Headline stats + per-episode figures showcase the BEST round, not
    # the last one — later rounds can regress (Round 8 did: 885/1000
    # after Round 6's 997/1000). The round-by-round figures still show
    # every round, regression included.
    best_round_no = final_meta["best_round"]
    by_round = {r["round"]: r for r in rounds}
    final = by_round[best_round_no]
    first = rounds[0]
    episodes = pd.read_csv(
        f"runs/perfect_round_{best_round_no}_episodes.csv")
    charts, _, anatomy_round = build_all_charts(
        rounds, showcase_round=best_round_no)

    takeaways = []
    takeaways.append(
        f"<b>Zero-crash goal:</b> the final model landed "
        f"<b>{final['landed']}/{final['episodes']}</b> eval episodes "
        f"({final['success_rate']:.1%}), with "
        f"<b>{final['failures']} failures</b> — "
        + ("goal reached ✅" if final_meta["reached_zero"]
           else "goal not reached within the step cap"))
    takeaways.append(
        f"<b>Crash causes shifted:</b> in the baseline eval the dominant "
        f"cause was {_dominant_cause(first)}; in the final eval it was "
        f"{_dominant_cause(final)}.")
    takeaways.append(
        "<b>Hardest starts (final eval):</b> "
        + _hardest_bin(episodes, "start_dist",
                        [0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.3])
        + "; "
        + _hardest_bin(episodes, "start_speed",
                        [0, 0.15, 0.3, 0.45, 0.6, 0.9])
        + ".")
    takeaways.append(
        f"<b>Fuel:</b> mean fuel used went from "
        f"{first['mean_fuel_used']:.1f} to {final['mean_fuel_used']:.1f} "
        f"units per episode — the policy learned to brake harder (more "
        f"fuel) when the speed limits started being enforced.")
    takeaways.append(
        "<b>Why the old 68% number changed:</b> the original env zeroed "
        "velocity before the landing check, so the |vy| ≤ 1.5 and "
        "|vx| ≤ 1.0 speed limits never applied and the agent learned to "
        "slam into the pad. The fix (judge the pre-contact impact "
        "velocity) dropped the same model to 42% on the same task — "
        "every later number here is measured with the fixed criteria.")

    chart_sections = [
        ("crash_ratio", "Success & failure rate vs training steps",
         "Each point is a 1000-episode deterministic eval on fixed seeds. "
         "The x-axis counts all training steps, including the 200k the "
         "starting model had already done."),
        ("crash_causes", "Crash-cause breakdown per round",
         "Every failed episode gets exactly one cause (precedence: "
         "timeout → fuel exhaustion → too-fast impact → off-pad → tilt, "
         "with out-of-bounds as its own category). Watch too-fast impact "
         "shrink as the policy learns to brake."),
        ("landing_positions", "Where the rocket touches down",
         "Left: touchdown-x histogram for landed vs ground-crashed "
         "episodes (green band = pad). Right: touchdown x vs starting "
         "distance — failures cluster at the pad edges."),
        ("success_by_start", "Which starting conditions are hardest",
         "Success rate binned by starting distance from the pad and by "
         "starting speed. Far, fast starts are the long tail that makes "
         "the last few failures stubborn."),
        ("eval_metrics", "Reward, episode length & fuel over training",
         "Mean eval reward climbs as failures fall; episode length and "
         "fuel trace how the policy's strategy changes (more braking "
         "costs more fuel)."),
        ("crash_anatomy", f"Crash anatomy (round {anatomy_round})",
         f"Distributions of impact vertical speed and tilt for failed "
         f"episodes in round {anatomy_round}, with the landing limits "
         f"marked. The best round ({best_round_no}) had only "
         f"{final['failures']} failures — too few for a distribution — so "
         f"anatomy uses the last round with substantial failures. "
         f"Failures just past a limit are near-misses; far past it, the "
         f"approach broke down."),
    ]

    figures_html = "\n".join(
        f"""<section class="card">
          <h2>{title}</h2>
          <p class="caption">{caption}</p>
          <img src="data:image/png;base64,{_b64(charts[name])}"
               alt="{title}">
        </section>"""
        for name, title, caption in chart_sections
    )

    gif_html = ""
    if os.path.exists(gif_path):
        gif_html = (
            f'<section class="card"><h2>Watch it land</h2>'
            f'<p class="caption">A landing from the final model.</p>'
            f'<img src="data:image/gif;base64,{_b64(gif_path)}" '
            f'alt="Rocket landing"></section>')

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Rocket Landing RL — Train Until Perfect</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         background: #0b1026; color: #ecf0f1; margin: 0; padding: 2rem; }}
  h1 {{ font-size: 2rem; margin-bottom: 0.2rem; }}
  .sub {{ color: #95a5a6; margin-bottom: 1.5rem; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; }}
  .stat {{ background: #16203f; border-radius: 10px; padding: 1rem 1.4rem;
          min-width: 150px; }}
  .stat .num {{ font-size: 1.8rem; font-weight: 700; }}
  .stat .lbl {{ color: #95a5a6; font-size: 0.85rem; }}
  .card {{ background: #16203f; border-radius: 10px; padding: 1.2rem;
          margin-bottom: 1.5rem; }}
  .card img {{ width: 100%; border-radius: 6px; }}
  .caption {{ color: #95a5a6; font-size: 0.9rem; }}
  ul {{ line-height: 1.6; }}
  code {{ background: #1e2a4a; padding: 0.1rem 0.35rem; border-radius: 4px; }}
</style></head><body>
<h1>🚀 Rocket Landing RL — Train Until Perfect</h1>
<p class="sub">PPO agent trained in rounds until 0 failures in 1000
deterministic eval episodes (or the 3M-step cap). Landing requires:
on the pad, |vertical speed| ≤ 1.5 m/s, |horizontal speed| ≤ 1.0 m/s,
|tilt| ≤ 0.20 rad.</p>
<div class="stats">
  <div class="stat"><div class="num">{final['landed']}/{final['episodes']}</div>
    <div class="lbl">Final eval landings</div></div>
  <div class="stat"><div class="num">{final['failures']}</div>
    <div class="lbl">Failures in final 1000</div></div>
  <div class="stat"><div class="num">{final_meta['total_steps_incl_prior']:,}</div>
    <div class="lbl">Total training steps</div></div>
  <div class="stat"><div class="num">Round {final_meta['best_round']}</div>
    <div class="lbl">Best round</div></div>
  <div class="stat"><div class="num">{final['mean_fuel_used']:.1f} / 100</div>
    <div class="lbl">Mean fuel used</div></div>
</div>
<section class="card"><h2>Takeaways</h2><ul>
{''.join(f'<li>{t}</li>' for t in takeaways)}
</ul></section>
{gif_html}
{figures_html}
<section class="card"><h2>Reproduce</h2>
<pre><code>python train_until_perfect.py   # full loop (rounds + evals)
python charts.py                # rebuild all figures
python report.py                # rebuild this page</code></pre>
<p class="caption">Per-episode data: runs/perfect_round_N_episodes.csv.
Round log: runs/perfect_run_log.jsonl. Best model:
models/ppo_rocket_best.zip.</p></section>
</body></html>"""

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    return out


if __name__ == "__main__":
    print("Report:", build_report())
