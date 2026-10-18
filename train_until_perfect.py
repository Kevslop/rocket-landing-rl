"""
train_until_perfect.py — Train PPO in rounds until 0 failures in 1000 eval
episodes (or until the step cap is hit).

Loop:
    1. Baseline-eval the starting model on 1000 fixed-seed episodes
       (round 0) — seeds are far from anything used in training.
    2. Each round: train `round_steps` more steps continuing from the
       current model (PPO.load), save a checkpoint, then re-evaluate
       deterministically on the SAME 1000 eval episodes.
    3. If failures hit 0, stop and keep that model as best. Otherwise
       keep training. The best checkpoint (fewest failures) is always
       saved to --out (default models/ppo_rocket_best.zip).

Hyperparameter schedule: rounds 1-3 use the original PPO settings
(lr 3e-4, entropy 0.01); later rounds switch to fine-tuning
(lr 1e-4, entropy 0.002) so the policy polishes instead of thrashing.

Honesty rules: the step cap (~3M new steps) is hard — if it is hit
without reaching 0 failures, the script stops and reports the best
failure count. Eval uses the same 1000 seeds every round (no
cherry-picking), and the landing criteria are the fixed, documented
ones (see env.py).

Usage:
    python train_until_perfect.py                        # full run
    python train_until_perfect.py --round-steps 2000 --max-new-steps 4000 \\
        --eval-episodes 20 --rounds-max 2               # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time

from env import RocketLandingEnv
from diagnostics import CRASH_CAUSES, run_eval_episodes, summarize


def apply_schedule(model, lr: float, ent_coef: float):
    """Change lr / entropy on a loaded PPO model.

    SB3 builds the LR schedule once at load time, so setting
    `learning_rate` alone is not enough — the schedule function must
    be rebuilt too. `ent_coef` is read directly at loss time, so a
    plain assignment works.
    """
    from stable_baselines3.common.utils import get_schedule_fn

    model.learning_rate = lr
    model.lr_schedule = get_schedule_fn(lr)
    model.ent_coef = ent_coef
    return model


def schedule_for_round(round_no: int):
    """(lr, ent_coef) for a training round. Round 0 = baseline eval."""
    if round_no <= 3:
        return 3e-4, 0.01
    return 1e-4, 0.002


class TrainingCurveCallback:
    """Logs per-episode training rows (round, cumulative step, reward,
    length, landed, crashed) to runs/perfect_training_curve.csv."""

    def __init__(self, csv_path: str, round_no: int):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Cb(BaseCallback):
            def __init__(inner):
                super().__init__()
                inner._ep_reward = 0.0
                inner._ep_len = 0
                inner._file = open(csv_path, "a", newline="")
                inner._writer = csv.writer(inner._file)
                if os.path.getsize(csv_path) == 0:
                    inner._writer.writerow(
                        ["round", "cum_steps", "reward", "length",
                         "landed", "crashed"])
                inner._round = round_no

            def _on_step(inner) -> bool:
                inner._ep_reward += float(inner.locals["rewards"][0])
                inner._ep_len += 1
                for i, done in enumerate(inner.locals["dones"]):
                    if done:
                        info = inner.locals["infos"][i]
                        inner._writer.writerow([
                            inner._round, inner.num_timesteps,
                            round(inner._ep_reward, 2), inner._ep_len,
                            int(info.get("landed", False)),
                            int(info.get("crashed", False)),
                        ])
                        inner._ep_reward = 0.0
                        inner._ep_len = 0
                return True

            def _on_training_end(inner) -> None:
                inner._file.close()

        self.cb = _Cb()

    def callback(self):
        return self.cb


def write_episodes_csv(rows, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["episode", "seed", "reward", "start_x", "start_y",
              "start_vx", "start_vy", "start_angle", "start_ang_vel",
              "start_speed", "start_dist", "impact_vx", "impact_vy",
              "impact_angle", "touchdown_x", "final_x", "final_y",
              "fuel_used", "fuel_left", "steps", "landed", "crashed",
              "truncated", "crash_cause", "outcome"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():
    p = argparse.ArgumentParser(
        description="Train PPO until 0 failures in 1000 eval episodes.")
    p.add_argument("--start-from", default="models/ppo_rocket.zip",
                   help="starting checkpoint (trained from scratch if missing)")
    p.add_argument("--prior-steps", type=int, default=200_000,
                   help="steps the starting model already trained (x-axis)")
    p.add_argument("--round-steps", type=int, default=400_000)
    p.add_argument("--max-new-steps", type=int, default=3_000_000,
                   help="hard cap on NEW training steps in this run")
    p.add_argument("--rounds-max", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=1000)
    p.add_argument("--eval-seed-base", type=int, default=1_000_000)
    p.add_argument("--out", default="models/ppo_rocket_best")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", default="runs/perfect_run_log.jsonl")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    train_env = Monitor(RocketLandingEnv(seed=args.seed))
    if os.path.exists(args.start_from):
        print(f"Resuming from {args.start_from}")
        model = PPO.load(args.start_from, env=train_env)
        print(f"Loaded model num_timesteps={model.num_timesteps}")
    else:
        print("No starting checkpoint — training from scratch.")
        model = PPO("MlpPolicy", train_env, learning_rate=3e-4,
                    n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                    vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs={"net_arch": [128, 128]},
                    seed=args.seed, verbose=0)

    def eval_round(round_no: int, cum_steps: int):
        t0 = time.time()
        rows = run_eval_episodes(model, args.eval_episodes,
                                 seed_base=args.eval_seed_base)
        s = summarize(rows)
        s.update({"round": round_no, "cum_steps": cum_steps,
                  "eval_seconds": round(time.time() - t0, 1)})
        write_episodes_csv(rows, f"runs/perfect_round_{round_no}_episodes.csv")
        return s

    log_f = open(args.log, "a")
    best_failures = None
    best_round = None
    new_steps = 0

    def log_round(s):
        log_f.write(json.dumps(s) + "\n")
        log_f.flush()
        causes = {k: v for k, v in s["crash_causes"].items() if v}
        print(f"Round {s['round']} | cum_steps {s['cum_steps']:,} | "
              f"landed {s['landed']}/{s['episodes']} "
              f"({s['success_rate']:.1%}) | failures {s['failures']} "
              f"| causes {causes} | mean reward {s['mean_reward']:.1f} "
              f"| fuel {s['mean_fuel_used']:.1f} "
              f"| eval {s['eval_seconds']}s", flush=True)

    # --- Round 0: baseline eval of the starting model ---
    s0 = eval_round(0, args.prior_steps)
    log_round(s0)
    best_failures, best_round = s0["failures"], 0

    # --- Training rounds ---
    round_no = 0
    reached_zero = s0["failures"] == 0
    while not reached_zero and new_steps < args.max_new_steps \
            and round_no < args.rounds_max:
        round_no += 1
        lr, ent = schedule_for_round(round_no)
        steps_this_round = min(args.round_steps,
                               args.max_new_steps - new_steps)
        print(f"--- Round {round_no}: training {steps_this_round:,} steps "
              f"(lr={lr}, ent={ent}) ---", flush=True)
        apply_schedule(model, lr, ent)
        cb = TrainingCurveCallback("runs/perfect_training_curve.csv",
                                   round_no).callback()
        t0 = time.time()
        model.learn(total_timesteps=steps_this_round, callback=cb,
                    reset_num_timesteps=False)
        train_secs = round(time.time() - t0, 1)
        new_steps += steps_this_round

        ckpt = f"models/perfect_round_{round_no}"
        model.save(ckpt)
        with open(ckpt + "_config.json", "w") as f:
            json.dump({"round": round_no, "steps_this_round": steps_this_round,
                       "cum_new_steps": new_steps, "lr": lr, "ent_coef": ent,
                       "eval_episodes": args.eval_episodes,
                       "eval_seed_base": args.eval_seed_base}, f, indent=2)

        s = eval_round(round_no, args.prior_steps + new_steps)
        s["train_seconds"] = train_secs
        s["lr"] = lr
        s["ent_coef"] = ent
        log_round(s)
        if s["failures"] < best_failures:
            best_failures, best_round = s["failures"], round_no
            shutil.copy(ckpt + ".zip", args.out + ".zip")
            shutil.copy(ckpt + "_config.json", args.out + "_config.json")
            print(f"  -> new best (round {round_no}); saved to {args.out}.zip",
                  flush=True)
        reached_zero = s["failures"] == 0

    log_f.close()

    final = {
        "reached_zero": reached_zero,
        "best_round": best_round,
        "best_failures": best_failures,
        "best_failures_per_1000": best_failures,
        "total_new_steps": new_steps,
        "total_steps_incl_prior": args.prior_steps + new_steps,
        "best_model": args.out + ".zip",
        "eval_episodes": args.eval_episodes,
        "eval_seed_base": args.eval_seed_base,
    }
    with open("runs/perfect_final.json", "w") as f:
        json.dump(final, f, indent=2)
    print("=== FINAL ===")
    print(json.dumps(final, indent=2))
    return final


if __name__ == "__main__":
    main()
