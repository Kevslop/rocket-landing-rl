# Rocket Landing RL 🚀

> **v2 is here — the 3D mission sim.** This repo now has two versions:
> - **[v2/](v2/) — current.** Full 3D takeoff → waypoint → landing mission with
>   engine loadouts (Raptor 3, Merlin 1D, BE-4, Rutherford + projected aerospike /
>   rotating-detonation concepts), moving landing pad, ±5% pad slopes, rain/snow/ice
>   weather, domain randomization, and an interactive SpaceX-style 3D demo
>   (watch AI, manual keyboard flight, slow-mo replay). See [v2/README.md](v2/README.md).
> - **v1 (below) — outdated.** The original 2D landing prototype (99.7% landing
>   success, 997/1000 evals). Kept for reference; all new work is in v2/.

Train a reinforcement-learning agent to land a 2D rocket on a pad — a
from-scratch spinoff built on open-source libraries (Gymnasium +
stable-baselines3). No code is copied from any existing rocket-landing
repo; the environment, physics, and reward shaping are original to this
project. Everything runs fully locally — no API keys, no hosted calls.

![landing](runs/eval_landing.gif)

## What it is

A 2D rigid-body rocket must descend from a randomised starting position
and land **softly, upright, on the landing pad**. The agent controls a
main engine (off / half / full) plus two side thrusters for attitude
control, with limited fuel. Episodes end on landing, crash, flying
off-screen, or timeout.

## Physics model (`env.py`)

| Quantity | Value |
|---|---|
| Gravity | 9.81 m/s² downward |
| Main thrust (full) | 22 m/s² along the rocket axis (can hover/climb) |
| Side-thruster torque | 6 rad/s² per thruster |
| Linear / angular drag | 0.10 /s, 0.30 /s |
| Fuel | 100 units; main engine 0.18/step at full throttle, RCS 0.03/step each |
| Timestep | 1/60 s, semi-implicit Euler integration |
| Wind | optional constant horizontal acceleration (`--wind`) |

State (8-D): `[x, y, vx, vy, angle, angular_velocity, leg_L, leg_R]`.
Actions (`MultiDiscrete([3, 2, 2])`): main engine off/half/full, left RCS
off/on, right RCS off/on. Ground contact is inelastic: touching down with
the base at y = 0 zeroes velocity. A landing counts when the rocket
touches down **on the pad**, with |vertical speed| ≤ 1.5 m/s,
|horizontal speed| ≤ 1.0 m/s, and |tilt| ≤ 0.20 rad (~11°).

## Reward design

Dense shaping so PPO learns quickly:

- `-0.30 × distance to pad centre` — always pulled toward the pad
- `-0.10 × (|vx| + |vy|)` — slow, controlled descent
- `-0.20 × |angle|` — stay near-vertical
- `-0.01` per step — finish promptly, don't hover forever
- `-0.02 × throttle`, `-0.005` per RCS fire — fuel thrift
- `+0.10 × (1 − |vy|/5)` when low, slow, and over the pad — approach bonus
- **+100** for a clean landing
- **−50 − 10·|vy| − 20·|angle|** for a crash (scales with impact severity)
- **−100** for flying off-screen, **−10** on timeout

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+, CPU-only PyTorch is fine (install via
`pip install torch --index-url https://download.pytorch.org/whl/cpu`
if you want to avoid the large CUDA wheels).

## Train

```bash
python train.py                     # default: 200k timesteps
python train.py --timesteps 500000  # longer run
python train.py --smoke             # quick smoke test (~4k steps)
python train.py --wind 1.5 --out models/windy
python train.py --load models/ppo_rocket.zip --timesteps 100000  # continue
```

Outputs in `models/`: `<name>.zip` (policy), `<name>_curve.csv`
(per-episode reward / length / landed / crashed), `<name>_config.json`.

## Evaluate

```bash
python evaluate.py                                   # 50 episodes
python evaluate.py --episodes 100 --gif runs/my_run.gif --seed 123
```

Prints success rate (landed on pad), crash rate, mean reward, mean fuel
used, mean episode length; saves a GIF of a landing attempt and appends a
JSON-lines summary to `runs/eval_summary.jsonl`.

## Train until perfect

```bash
python train_until_perfect.py   # rounds of 400k steps + 1000-episode evals
```

Trains PPO in rounds until the agent scores **0 failures in 1000
deterministic eval episodes** (fixed seeds, far from training seeds),
or until the 3M-new-step cap. Each round saves a checkpoint
(`models/perfect_round_N.zip`); the best checkpoint (fewest failures)
is kept as `models/ppo_rocket_best.zip`. Rounds 1–3 use lr 3e-4 /
entropy 0.01; later rounds fine-tune at lr 1e-4 / entropy 0.002.
Per-round summaries go to `runs/perfect_run_log.jsonl`, per-episode
diagnostics (starting conditions, impact velocity/tilt, touchdown x,
crash cause) to `runs/perfect_round_N_episodes.csv`.

## Analytics & report

```bash
python charts.py    # 6 figures -> runs/charts/
python report.py    # showcase page -> runs/report.html
```

Open **[runs/report.html](runs/report.html)** for the full story:
success vs training steps, crash-cause breakdown per round,
touchdown-position distribution, hardest starting conditions, eval
metrics, and crash anatomy — all from real logged data.

## Results — the zero-crash run

**Goal: 0 failures in 1000 eval episodes. Result: not reached —
best was 3 failures / 1000 (99.7% success).** Full run, 3.2M total
training steps (200k prior + 3M new), 1000-episode deterministic
evals per round:

| Round | Cum. steps | Landed / 1000 | Failures | Dominant crash cause |
|---|---|---|---|---|
| 0 (baseline) | 200,704 | 435 | 565 | too-fast impact (563) |
| 1 | 600,704 | 897 | 103 | off-pad (75), too-fast (27) |
| 2 | 1,000,704 | 881 | 119 | off-pad (119) |
| 3 | 1,400,704 | 984 | 16 | off-pad (11), too-fast (5) |
| 4 | 1,800,704 | 991 | 9 | off-pad (9) |
| 5 | 2,200,704 | 990 | 10 | off-pad (7), too-fast (3) |
| **6 (best)** | **2,600,704** | **997** | **3** | **too-fast (2), off-pad (1)** |
| 7 | 3,000,704 | 978 | 22 | off-pad (22) |
| 8 | 3,200,704 | 885 | 115 | off-pad (115) |

Best model: `models/ppo_rocket_best.zip` (Round 6). Mean reward 85.6,
mean fuel 4.0/100, mean episode length 37 steps. Later rounds
regressed — PPO is noisy at this precision, and the last few failures
are stubborn edge cases (far, fast starts landing just outside the
pad). Round 8 was re-run from the Round 7 checkpoint after the
original process died mid-round. The run log and per-episode CSVs
are in `runs/` for anyone who wants to dig in.

> **Correction to the earlier numbers:** the first version of this
> repo reported 68% success (34/50) — but the env had a bug: velocity
> was zeroed *before* the landing check, so the |vy| ≤ 1.5 and
> |vx| ≤ 1.0 speed limits never applied and the agent learned to slam
> into the pad. Fixing it (judge the pre-contact impact velocity)
> dropped the same model to 43.5% on 1000 episodes. All numbers
> above use the fixed criteria. Physics constants and landing
> thresholds are unchanged — only the bug is fixed.

## Project layout

```
env.py                  Gymnasium environment (physics, reward, termination, diagnostics)
render.py               Matplotlib renderer: rocket, flame, pad, trace, telemetry (headless)
train.py                PPO training + per-episode curve logging
train_until_perfect.py  Round-based training until 0 failures / 1000 evals
diagnostics.py          Deterministic eval with per-episode diagnostics + crash taxonomy
charts.py               6 analytics figures -> runs/charts/
report.py               Showcase page -> runs/report.html
evaluate.py             Evaluation + landing GIF (+ --diag-csv)
requirements.txt        pinned open-source deps
```

## Ideas for extension

- **Continuous actions** — swap `MultiDiscrete` for a `Box` action space
  (throttle 0–1, torque −1–1) and use SAC instead of PPO.
- **Wind gusts** — time-varying wind instead of a constant value.
- **Moving pad** — a pad that drifts, or a sea-barge that bobs.
- **Curriculum** — start episodes closer to the pad, then widen the spawn.
- **Fuel-optimal landing** — add a landing bonus that scales with fuel left.
- **3D** — extend the state/action space to a full 3D rigid body.
- **Human play mode** — keyboard control (arrow keys) to fly it yourself.
- **Compare algorithms** — benchmark PPO vs A2C vs SAC on the same env.

## License

MIT — use it, fork it, extend it.
