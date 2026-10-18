# v2 — 3D Takeoff + Landing Mission with Engine Loadouts

An interactive 3D rocket mission simulator. A single-engine hopper lifts off
from a launch pad, flies to a randomized hover waypoint, then lands on a
**moving, sloped landing pad** — in randomized **weather** (clear / rain /
snow / ice). You pick the **engine loadout** (real engine + standard or
experimental propulsion tech) and the trained RL pilot flies the mission;
or you fly it yourself with the keyboard. The HUD is styled after SpaceX's
Starship landing broadcasts.

## The question it answers

> *Can we take off and land efficiently with this loadout?*

Pick any engine + propulsion-tech combo and the sim reports per-mission:
**outcome** (landed / crashed + reason), **fuel used**, **flight time**, and
waypoint capture. `evaluate_mission.py` runs the same check across all 16
loadouts and writes `v2/eval_mission.json`.

## Engine loadouts

**Real engines** (published specs — see `engines.py` for sources):

| Engine | Maker | Cycle | SL thrust | Isp (SL/vac) | Throttle | Gimbal | Mass | Pc |
|---|---|---|---|---|---|---|---|---|
| Raptor 3 | SpaceX | FFSC methalox | 2.75 MN | 327 / 350 s | 40–100% | ±8°¹ | 1,525 kg | 350 bar |
| Merlin 1D | SpaceX | Gas-gen kerolox | 845 kN | 282 / 311 s | 57–100% | ±5°¹ | 470 kg | 97 bar |
| BE-4 | Blue Origin | ORSC methalox | 2.46 MN | 340 / 340 s | 40–100% | ±5° | 5,400 kg | 140 bar |
| Rutherford | Rocket Lab | Electric-pump kerolox | 25 kN | 311 / 343 s | 50–100%¹ | ±5°¹ | 35 kg | 100 bar¹ |

¹ Estimated where the manufacturer has not published a value.

**Propulsion tech** (applied to any engine):

| Tech | Status | Effect (projected) |
|---|---|---|
| Standard bell nozzle | Flight-proven | Baseline |
| Aerospike nozzle | **PROJECTED** | Altitude compensation: vacuum Isp at *all* altitudes; +8% engine mass, −2% thrust |
| Rotating detonation combustor (RDRE) | **PROJECTED** | +15% Isp, +10% thrust (vendor claim), min throttle 60%, +12% mass |
| RDRE + aerospike | **PROJECTED** (concept) | Both effects combined |

Basis for the projections: LEAP 71's 200 kN 3D-printed methalox aerospike
(XRA-2E5, hot-fire campaign 2026; 20 kN aerospike hot-fired Dec 2025) and
Pangea Aerospace's MERLIn aerospike program — **no aerospike engine has flown
in space as of 2026**. Venus Aerospace flight-tested a high-thrust RDRE in
May 2025 (~2,000 lbf, 7 s flight) claiming ~15% better efficiency than
conventional engines, and raised a $91M Series B in July 2026; NASA
demonstrated an RDRE on the ground in 2022. The detonation-frequency readout
(~2.4 kHz) and RDRE throttle/mass penalties are engineering estimates.

The hopper is sized per loadout (single engine, liftoff T/W = 1.35), so a
Raptor 3 vehicle masses ~208 t while a Rutherford one masses ~1.9 t — engine
choice changes vehicle performance meaningfully.

## Mission & environment (`mission_env.py`, Gymnasium API)

- **Phase 0 — Ascent:** take off, capture a hover waypoint (azimuth 0–360°,
  20–40 m out, 45–60 m up; capture radius 10 m, speed < 8 m/s).
- **Phase 1 — Landing:** land on the landing pad — randomized position
  (15–40 m from launch), random incline up to ±5% grade (in the observation
  vector; touchdown attitude is judged against the pad normal), pad friction
  per weather. Touchdown horizontal speed causes sliding
  (d = v²/2μg); sliding off the pad = crash. Ice (μ≈0.05) demands a
  near-zero-speed touchdown.
- **Weather** (randomized in training, selectable in the demo): clear, rain
  (+gusts, −2% Isp, reduced visibility), snow (+gusts, −3% Isp), ice (slick pad).
- **Domain randomization:** engine/tech loadout, vehicle mass ±, fuel load
  85–100%, waypoint/landing-pad geometry, pad slope, wind base + OU gusts,
  weather, initial tilt rates.

One loadout-conditioned PPO policy (28-D obs incl. phase, waypoint, pad
pose/slope/friction, weather id, and loadout descriptors) flies all 16
loadouts.

## Train

```bash
cd v2
# curriculum: 1) steer+waypoint  2) precision landing  3) full mission
../.venv/bin/python train_mission.py --spawn air --timesteps 600000 --run-name stage1a
../.venv/bin/python train_mission.py --spawn landing --timesteps 600000 --run-name stage1b \
    --warmstart models/stage1a.zip
../.venv/bin/python train_mission.py --spawn ground --timesteps 1500000 --run-name best_mission \
    --warmstart models/stage1b.zip
# best -> v2/models/best_mission.zip
```

**Training notes (honest):** the full takeoff → waypoint → landing mission is
a hard exploration problem for vanilla PPO. With a 3-stage curriculum the
policy reliably learns to take off and steer to the waypoint (~58% capture),
but precision landing on the 7 m pad stayed at ~10% for pure PPO — the agent
learns that hovering beats risking a crash. Behavioral cloning from a
scripted PD controller (`scripted.py`) reaches 55% landing, confirming the
task is feasible. The interactive demo therefore flies with the scripted PD
controller by default (`--pilot scripted`); pass `--pilot ppo --model ...` to
fly any trained PPO checkpoint instead. `scripted.py` also serves as a
reference controller and BC demonstration source (`bc.py`).

## Evaluate (all loadouts)

```bash
../.venv/bin/python evaluate_mission.py --episodes 60 --pilot scripted
# -> v2/eval_mission.json  (success rate, fuel used, flight time per loadout)
```

With the scripted pilot (40 episodes/loadout, weather randomized): landing
success 8–40% depending on loadout (RDRE's extra thrust helps most),
fuel used 30–52% of load, flight time 26–48 s. The Rutherford hopper is the
least efficient (tiny engine, long flight). See `eval_mission.json` for the
full table — this is the "can we take off and land efficiently?" report.

## Interactive demo

```bash
cd v2
../.venv/bin/python server.py            # scripted pilot (default)
../.venv/bin/python server.py --pilot ppo --model models/best_mission.zip
# open http://localhost:8123
```

- **Watch AI** — the pilot flies the selected loadout (scripted PD by default).
- **Loadout picker** (left) — engine + propulsion tech + weather, then *Fly loadout*.
- **Fly manual** — W/S pitch, A/D roll (camera-relative), R/F throttle, X cutoff.
- **Speed** — 0.25× / 0.5× / 1× (default 0.5× so the landing is watchable).
- **Randomize** — new mission geometry/weather; **Auto** — continuous flights;
  **Replay** — slow-mo replay of the last flight; camera cycles follow/pad/tower.
- HUD: mission clock, phase, altitude, vertical/horizontal speed, pad
  distance/bearing/slope, propellant + throttle + chamber-pressure bars,
  engine lights, detonation frequency (RDRE), wind, T/W, weather badge.

All physics and the PPO policy run server-side (Python); the browser renders
Three.js (vendored, no build step). No internet needed after setup.

## Files

| File | What |
|---|---|
| `engines.py` | Engine presets, experimental tech, loadout resolution |
| `mission_env.py` | Takeoff→waypoint→landing Gymnasium env (28-D obs) |
| `train_mission.py` | PPO training, curriculum (air/landing/ground spawns) |
| `bc.py` | Behavioral cloning from the scripted controller |
| `scripted.py` | Reference PD controller (demo pilot, BC demos) |
| `evaluate_mission.py` | Per-loadout efficiency report |
| `server.py` | WebSocket sim server + static file server |
| `demo/` | Three.js frontend (SpaceX-style HUD) |
| `env3d.py` / `train.py` / `evaluate.py` | Earlier landing-only 3D prototype (superseded by the mission) |
