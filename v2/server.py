"""
server.py — Interactive v2 demo backend (mission edition).

Runs the take-off -> waypoint -> landing mission sim with a loadout
(engine preset + standard/projected propulsion tech, see engines.py),
streaming state to the Three.js frontend over WebSocket. The single
loadout-conditioned PPO policy flies every loadout; manual mode lets the
user fly by keyboard.

Usage:
    cd v2 && ../.venv/bin/python server.py [--model models/best_mission.zip]
                  [--http-port 8123] [--ws-port 8124]

Then open http://localhost:8123. One start command, no build step.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import math
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import websockets

    from stable_baselines3 import PPO

    from engines import ENGINES, G0, TECHS, WEATHER, WEATHER_IDS, resolve_loadout
    from mission_env import LoadoutMissionEnv
except ImportError as e:  # pragma: no cover
    sys.exit(f"missing dependency: {e} (pip install -r requirements.txt)")


class Sim:
    """Single-session mission sim. Physics ticks at 20 Hz * speed_scale."""

    SPEEDS = [0.25, 0.5, 1.0]

    def __init__(self, model, pilot="ppo"):
        self.model = model
        self.pilot = pilot  # "ppo" or "scripted"
        self.engine_id = "raptor3"
        self.tech_id = "standard"
        self.weather_sel = "auto"  # or a weather id
        self.env = LoadoutMissionEnv(
            engine_id=self.engine_id, tech_id=self.tech_id,
            weather=self.weather_sel, seed=1234,
        )
        self.mode = "ai"
        self.auto = False
        self.speed = 0.5  # default: watchable slow-mo
        self.manual = {"throttle": 0.0, "tilt_x": 0.0, "tilt_z": 0.0}
        self.obs, _ = self.env.reset()
        self.done = False
        self.outcome = None
        self.report = None
        self.traj = []
        self._cooldown = 0.0
        self._ep = 0

    def set_loadout(self, engine_id, tech_id, weather):
        if engine_id in ENGINES:
            self.engine_id = engine_id
        if tech_id in TECHS:
            self.tech_id = tech_id
        if weather in WEATHER_IDS or weather == "auto":
            self.weather_sel = weather
        self.env = LoadoutMissionEnv(
            engine_id=self.engine_id, tech_id=self.tech_id,
            weather=self.weather_sel, seed=2000 + self._ep,
        )
        self.new_episode()

    def new_episode(self, mode=None, seed=None):
        if mode in ("ai", "manual"):
            self.mode = mode
        self._ep += 1
        self.obs, _ = self.env.reset(
            seed=seed if seed is not None else 1000 + self._ep)
        self.done = False
        self.outcome = None
        self.report = None
        self.traj = []
        self._cooldown = 0.0
        self._record()

    def _record(self):
        e = self.env
        wind = e.wind_base + e.gust
        self.traj.append({
            "t": round(e.t, 2),
            "pos": [round(v, 2) for v in e.pos],
            "tilt": [round(v, 3) for v in e.tilt],
            "throttle": round(e.throttle, 3),
            "fuel": round(e.fuel / e.loadout["fuel_mass"], 3),
            "phase": e.phase,
        })

    def step(self):
        if self.done:
            if self.auto:
                self._cooldown -= 1 / 20
                if self._cooldown <= 0:
                    self.new_episode()
            return self._state_msg()
        if self.mode == "ai":
            if self.pilot == "scripted":
                from scripted import scripted_action
                action = scripted_action(self.env)
            elif self.model is not None:
                action, _ = self.model.predict(self.obs, deterministic=True)
            else:
                action = np.zeros(3, dtype=np.float32)
        else:
            m = self.manual
            action = np.array([
                m["throttle"] * 2 - 1,
                m["tilt_x"] / self.env.loadout["gimbal_rad"],
                m["tilt_z"] / self.env.loadout["gimbal_rad"],
            ], dtype=np.float32)
        self.obs, _, term, trunc, info = self.env.step(action)
        self.done = bool(term or trunc)
        if self.done:
            self.outcome = info.get("outcome")
            self.report = {
                "outcome": self.outcome,
                "reason": info.get("reason"),
                "fuel_used_pct": round(info["fuel_used_frac"] * 100, 1),
                "flight_time": round(info["flight_time"], 1),
                "waypoint_reached": info["waypoint_reached"],
            }
            self._cooldown = 2.5
        self._record()
        return self._state_msg()

    def _state_msg(self):
        e = self.env
        lo = e.loadout
        wind = e.wind_base + e.gust
        # chamber pressure follows throttle with a little noise
        pc = lo["pc_bar"] * (0.25 + 0.75 * e.throttle)
        det = None
        if lo["det_freq_khz"]:
            det = lo["det_freq_khz"] * (0.97 + 0.06 * e.throttle)
        msg = {
            "type": "state",
            "t": round(e.t, 2),
            "pos": [float(v) for v in e.pos],
            "vel": [float(v) for v in e.vel],
            "tilt": [float(v) for v in e.tilt],
            "throttle": float(e.throttle),
            "fuel": float(e.fuel / lo["fuel_mass"]),
            "wind": [float(v) for v in wind],
            "mass_t": round((lo["dry_mass"] + e.fuel) / 1000, 1),
            "done": self.done,
            "outcome": self.outcome,
            "mode": self.mode,
            "phase": e.phase,
            "waypoint": [float(v) for v in e.waypoint],
            "pad": {
                "x": float(e.pad[0]), "y": float(e.pad[1]),
                "r": e.PAD_R,
                "slope": [float(v) for v in e.slope],
                "mu": round(float(e.mu), 2),
            },
            "weather": e.weather_id,
            "engine": {
                "id": self.engine_id,
                "name": lo["engine_name"],
                "maker": lo["engine_maker"],
                "tech": lo["tech_name"],
                "projected": lo["projected"],
                "pc_bar": round(pc, 1),
                "det_khz": round(det, 2) if det else None,
                "tw": round(lo["thrust_sl"] / ((lo["dry_mass"] + e.fuel) * G0), 2),
                "scale": round((lo["wet_mass"] / 11000) ** (1 / 3), 2),
                "gimbal_deg": round(math.degrees(lo["gimbal_rad"]), 1),
            },
            "speed": self.speed,
        }
        if self.done:
            msg["trajectory"] = self.traj[::2]
            msg["report"] = self.report
        return msg

    def loadout_list_msg(self):
        return {
            "type": "loadouts",
            "engines": [
                {"id": k, "name": v["name"], "maker": v["maker"],
                 "blurb": f"{v['thrust_sl']/1e6:.2f} MN, Isp {v['isp_sl']}/{v['isp_vac']}s, "
                          f"throttle {int(v['throttle_min']*100)}-100%"}
                for k, v in ENGINES.items()
            ],
            "techs": [
                {"id": k, "name": v["name"], "projected": v["projected"],
                 "blurb": v["blurb"]}
                for k, v in TECHS.items()
            ],
            "weather": [{"id": k, "name": v["name"]} for k, v in WEATHER.items()],
        }


async def sim_loop(sim: Sim, clients: set):
    TICK = 1 / 20
    while True:
        await asyncio.sleep(TICK / max(sim.speed, 0.05))
        msg = sim.step()
        if clients:
            payload = json.dumps(msg)
            dead = set()
            for ws in clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.add(ws)
            clients -= dead


async def handle_client(sim: Sim, clients: set, ws):
    clients.add(ws)
    try:
        await ws.send(json.dumps({
            "type": "hello", "has_model": sim.model is not None,
            "mode": sim.mode, "speed": sim.speed,
            "engine": sim.engine_id, "tech": sim.tech_id,
            "weather": sim.weather_sel,
        }))
        await ws.send(json.dumps(sim.loadout_list_msg()))
        async for raw in ws:
            try:
                cmd = json.loads(raw)
            except Exception:
                continue
            c = cmd.get("cmd")
            if c == "new_episode":
                sim.new_episode(mode=cmd.get("mode"), seed=cmd.get("seed"))
            elif c == "set_loadout":
                sim.set_loadout(cmd.get("engine", sim.engine_id),
                                cmd.get("tech", sim.tech_id),
                                cmd.get("weather", sim.weather_sel))
            elif c == "manual":
                g = sim.env.loadout["gimbal_rad"]
                sim.manual = {
                    "throttle": float(np.clip(cmd.get("throttle", 0), 0, 1)),
                    "tilt_x": float(np.clip(cmd.get("tilt_x", 0), -g, g)),
                    "tilt_z": float(np.clip(cmd.get("tilt_z", 0), -g, g)),
                }
            elif c == "speed":
                s = float(cmd.get("value", 0.5))
                if s in Sim.SPEEDS:
                    sim.speed = s
            elif c == "auto":
                sim.auto = bool(cmd.get("on"))
    finally:
        clients.discard(ws)


def serve_http(port: int, directory: str):
    handler = functools.partial(SimpleHTTPRequestHandler, directory=directory)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def main_async(args):
    model = None
    pilot = args.pilot
    if pilot == "ppo":
        if os.path.exists(args.model):
            print("loading policy:", args.model, flush=True)
            model = PPO.load(args.model)
        else:
            print("no trained model at", args.model, "- manual mode only",
                  flush=True)
    else:
        print("pilot: scripted PD controller", flush=True)
    sim = Sim(model, pilot=pilot)
    clients: set = set()
    async with websockets.serve(
        functools.partial(handle_client, sim, clients), "127.0.0.1", args.ws_port
    ):
        await sim_loop(sim, clients)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "best_mission.zip"))
    ap.add_argument("--pilot", default="scripted", choices=["scripted", "ppo"],
                    help="AI pilot: scripted PD controller (reliable) or PPO policy")
    ap.add_argument("--http-port", type=int, default=8123)
    ap.add_argument("--ws-port", type=int, default=8124)
    args = ap.parse_args()

    serve_http(args.http_port, os.path.join(HERE, "demo"))
    print(f"demo at http://localhost:{args.http_port} (ws {args.ws_port})", flush=True)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
