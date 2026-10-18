"""
mission_env.py — Takeoff + landing mission environment (Gymnasium API).

Original code for this project. A single-engine hopper sized around a
"loadout" (real engine preset + standard/projected propulsion tech, see
engines.py):

    Phase 0 (ASCENT):  lift off from the launch pad, fly to a randomized
                       hover waypoint (azimuth / distance / altitude vary).
    Phase 1 (LANDING): land on a landing pad whose position is randomized
                       each episode (bounded area) and which has a random
                       incline of up to +/-5% grade.

Touchdown is modeled with pad friction: touchdown horizontal speed makes the
vehicle slide (slide distance = v^2 / (2*mu*g)); sliding off the pad is a
crash. Ice (mu=0.05) therefore demands a near-zero-speed touchdown. The
attitude requirement is measured against the pad's tilted normal, so the
agent must adapt to the slope.

Weather (clear/rain/snow/ice) changes pad friction, wind gusts, thrust
efficiency, and (in the demo) visibility. The policy is conditioned on the
loadout and environment parameters via the observation vector, so ONE policy
flies all 16 loadouts.

Observation (28 floats): vehicle state + wind + throttle + phase + waypoint +
landing-pad position/slope/friction + weather id + loadout descriptors.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    raise ImportError("gymnasium is required: pip install -r v2/requirements.txt")

from engines import ENGINES, G0, TECHS, WEATHER, WEATHER_IDS, resolve_loadout


class LoadoutMissionEnv(gym.Env):
    """Takeoff -> waypoint -> precision landing on a moving, sloped pad."""

    metadata = {"render_modes": ["rgb_array"]}

    DT = 0.05
    MAX_STEPS = 1500  # 75 s
    # PAD_R is now an instance attribute (curriculum); default 7.0 m

    # waypoint randomization (varied approaches)
    WP_DIST = (20.0, 40.0)
    WP_ALT = (45.0, 60.0)
    WP_CAP_R = 10.0
    WP_CAP_V = 8.0

    # landing pad randomization
    LAND_DIST = (15.0, 40.0)
    SLOPE_MAX = 0.05  # +/-5% grade

    # landing criteria
    MAX_VZ = 7.0
    MAX_VXY = 5.0
    MAX_ALIGN = 0.30  # rad between thrust axis and pad normal

    TILT_K = 6.0
    TILT_D = 4.5
    DRAG_REF = 1400.0  # N per m/s at 11 t reference mass
    DRAG_REF_MASS = 11000.0
    WIND_BASE = 6.0
    WIND_GUST = 4.0
    WIND_TAU = 2.5

    def __init__(self, engine_id="raptor3", tech_id="standard",
                 weather="auto", seed=None, randomize_loadout=False,
                 spawn="ground", pad_r=7.0):
        super().__init__()
        self.engine_id = engine_id
        self.tech_id = tech_id
        self.weather_mode = weather  # "auto" or a WEATHER id
        self.randomize_loadout = randomize_loadout
        # spawn: "ground"  = full takeoff mission;
        #        "air"     = curriculum: start airborne near the waypoint,
        #                    isolating the steer-and-capture problem
        #        "landing" = curriculum: start airborne near the landing pad
        #                    in phase 1, isolating the precision landing
        self.spawn = spawn
        self.PAD_R = pad_r  # curriculum: start large, shrink to 7.0
        self._apply_loadout(engine_id, tech_id)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(28,),
                                            dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._n_resets = 0
        self.reset(seed=seed)

    def _apply_loadout(self, engine_id, tech_id):
        self.engine_id = engine_id
        self.tech_id = tech_id
        self.loadout = resolve_loadout(engine_id, tech_id)
        lo = self.loadout
        # per-loadout derived constants
        self.drag_k = self.DRAG_REF * (lo["wet_mass"] / self.DRAG_REF_MASS) ** (2 / 3)

    # ------------------------------------------------------------------ api
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        r = self._rng
        lo = self.loadout

        # --- loadout resampling (training coverage over all combos) ------
        if self.randomize_loadout and self._n_resets > 0 and r.random() < 0.3:
            eng = r.choice(list(ENGINES))
            tech = r.choice(list(TECHS))
            self._apply_loadout(eng, tech)
            lo = self.loadout
        self._n_resets += 1

        # --- mission geometry (varied approaches) ------------------------
        az = r.uniform(0, 2 * math.pi)
        d = r.uniform(*self.WP_DIST)
        self.waypoint = np.array([
            d * math.cos(az), d * math.sin(az), r.uniform(*self.WP_ALT)])
        az2 = r.uniform(0, 2 * math.pi)
        d2 = r.uniform(*self.LAND_DIST)
        self.pad = np.array([d2 * math.cos(az2), d2 * math.sin(az2)])
        self.slope = r.uniform(-self.SLOPE_MAX, self.SLOPE_MAX, size=2)
        n = np.array([-self.slope[0], -self.slope[1], 1.0])
        self.pad_normal = n / np.linalg.norm(n)

        # --- vehicle ----------------------------------------------------
        self.fuel = lo["fuel_mass"] * r.uniform(0.85, 1.0)
        if self.spawn == "air":
            # curriculum: start airborne near the waypoint
            self.pos = self.waypoint + r.uniform([-15, -15, -5], [15, 15, 10])
            self.pos[2] = max(10.0, self.pos[2])
            self.vel = r.uniform(-3, 3, size=3)
            self.airborne = True
            self.phase = 0
            self.wp_reached = False
        elif self.spawn == "landing":
            # curriculum: start airborne near the pad, already in phase 1
            self.pos = np.array([
                self.pad[0] + r.uniform(-15, 15),
                self.pad[1] + r.uniform(-15, 15),
                r.uniform(20, 30)])
            self.vel = r.uniform(-3, 3, size=3)
            self.airborne = True
            self.phase = 1
            self.wp_reached = True
        else:
            self.pos = np.array([0.0, 0.0, 0.0])
            self.vel = np.zeros(3)
            self.airborne = False
            self.phase = 0
            self.wp_reached = False
        self.tilt = np.zeros(2)
        self.tilt_vel = np.zeros(2)
        self.throttle = 0.0

        # --- weather ------------------------------------------------------
        if self.weather_mode == "auto":
            self.weather_id = r.choice(
                WEATHER_IDS, p=[0.50, 0.20, 0.15, 0.15])
        else:
            self.weather_id = self.weather_mode
        w = WEATHER[self.weather_id]
        self.mu = w["mu"] * r.uniform(0.9, 1.1)
        self.isp_mult = w["isp_mult"]
        w_ang = r.uniform(0, 2 * math.pi)
        w_spd = r.uniform(0, self.WIND_BASE)
        self.wind_base = np.array(
            [w_spd * math.cos(w_ang), w_spd * math.sin(w_ang), 0.0])
        self.gust = np.zeros(3)
        self.gust_sigma = r.uniform(0.3, 1.0) * (
            self.WIND_GUST + w["gust_add"])

        self.t = 0.0
        self.steps = 0
        self._prev_pot = None
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        lo = self.loadout
        dt = self.DT
        r = self._rng

        # throttle maps into [throttle_min, 1]; biased so action 0 ~= hover
        thr_cmd = float(np.clip(0.85 + 0.35 * action[0],
                                lo["throttle_min"], 1.0))
        tilt_cmd = np.clip(action[1:], -1.0, 1.0) * lo["gimbal_rad"]

        # --- wind ---------------------------------------------------------
        theta = dt / self.WIND_TAU
        self.gust += -theta * self.gust + self.gust_sigma * math.sqrt(
            theta) * r.normal(size=3)
        self.gust[2] *= 0.3
        wind = self.wind_base + self.gust

        # --- propulsion ----------------------------------------------------
        if lo["altitude_compensation"]:
            isp_eff = lo["isp_vac"]  # aerospike: vacuum Isp at every altitude
        else:
            # bell nozzle: sea-level Isp low down, blending to vacuum Isp
            f = min(1.0, self.pos[2] / 8000.0)
            isp_eff = lo["isp_sl"] + (lo["isp_vac"] - lo["isp_sl"]) * f
        isp_eff *= self.isp_mult  # weather penalty (rain/snow)

        if self.fuel <= 0:
            self.throttle = 0.0
        else:
            self.throttle += np.clip(thr_cmd - self.throttle, -dt * 3, dt * 3)
            self.throttle = float(np.clip(self.throttle, 0.0, 1.0))
            mdot = self.throttle * lo["thrust_sl"] / (isp_eff * G0)
            self.fuel = max(0.0, self.fuel - mdot * dt)
        mass = lo["dry_mass"] + self.fuel

        # --- tilt servo ------------------------------------------------------
        self.tilt_vel += (self.TILT_K * (tilt_cmd - self.tilt)
                          - self.TILT_D * self.tilt_vel) * dt
        self.tilt += self.tilt_vel * dt

        tx, ty = self.tilt
        axis = np.array([math.sin(tx), math.sin(ty),
                         math.cos(tx) * math.cos(ty)])
        axis /= np.linalg.norm(axis)

        # --- forces ----------------------------------------------------------
        thrust = axis * (self.throttle * lo["thrust_sl"])
        gravity = np.array([0.0, 0.0, -mass * G0])
        drag = self.drag_k * (wind - self.vel)
        acc = (thrust + gravity + drag) / mass
        self.vel += acc * dt
        self.pos += self.vel * dt
        if self.pos[2] < 0.0:
            self.pos[2] = 0.0  # rest on the pad; touchdown judged when airborne
            if self.vel[2] < 0:
                self.vel[2] = 0.0
        self.t += dt
        self.steps += 1
        if self.pos[2] > 3.0 and not self.airborne:
            self.airborne = True
            liftoff_bonus = 10.0
        else:
            liftoff_bonus = 0.0

        # --- mission logic -----------------------------------------------------
        # potential shaping toward the active target; speed is only
        # penalized when close (so the agent isn't punished for flying fast
        # toward a distant target)
        target = self.waypoint if self.phase == 0 else np.array(
            [self.pad[0], self.pad[1], 0.0])
        dist = float(np.linalg.norm(self.pos - target))
        speed = float(np.linalg.norm(self.vel))
        reward = -0.002 - 0.005 * self.throttle + liftoff_bonus
        # NOTE: horizontal distance is weighted 3x per meter vs vertical.
        # The waypoint is high above, so a plain 3D distance makes "climb"
        # the whole solution and the agent never learns to steer sideways.
        d_xy = float(np.linalg.norm(self.pos[:2] - target[:2]))
        dz = abs(self.pos[2] - target[2])
        pot = -(d_xy / 20.0 + dz / 60.0)
        if dist < 15.0:
            pot -= (speed / 25.0) * (1.0 - dist / 15.0)
        if self._prev_pot is not None:
            reward += (pot - self._prev_pot) * 5.0
        self._prev_pot = pot

        # dense steering signal: reward HORIZONTAL velocity toward the target
        # (the 3D version is a trap: the waypoint is above, so the agent is
        # rewarded for climbing straight up without ever steering)
        to_xy = target[:2] - self.pos[:2]
        d_xy = float(np.linalg.norm(to_xy))
        v_xy = float(np.linalg.norm(self.vel[:2]))
        if d_xy > 1.0 and v_xy > 0.5:
            align_xy = float(np.dot(self.vel[:2], to_xy / d_xy) / v_xy)
            reward += 0.50 * align_xy * min(1.0, d_xy / 15.0)

        # final approach: in landing phase, penalize horizontal distance to
        # the pad directly (not just via potential differences) and punish
        # horizontal speed when close, so the agent centers and slows down
        if self.phase == 1:
            d_pad = float(np.linalg.norm(self.pos[:2] - self.pad))
            reward -= 0.05 * (d_pad / 20.0)
            if d_pad < 15.0:
                reward -= 0.10 * (v_xy / 5.0)

        # discourage blasting past the target altitude
        reward -= 0.10 * max(0.0, abs(self.vel[2]) - 10.0)
        if self.phase == 0 and self.pos[2] > self.waypoint[2] + 15.0:
            reward -= 0.05  # climbed way past the waypoint: come back down

        if self.phase == 0 and not self.wp_reached:
            if dist < self.WP_CAP_R and speed < self.WP_CAP_V:
                self.wp_reached = True
                self.phase = 1
                reward += 100.0
                self._prev_pot = None  # re-anchor shaping to the pad

        # --- termination ---------------------------------------------------------
        terminated, truncated, outcome, reason = False, False, None, None
        fuel_frac = self.fuel / lo["fuel_mass"]

        if self.pos[2] <= 0.0 and self.airborne:
            self.pos[2] = 0.0
            terminated = True
            dx, dy = self.pos[0] - self.pad[0], self.pos[1] - self.pad[1]
            dist_xy = math.hypot(dx, dy)
            vxy = math.hypot(self.vel[0], self.vel[1])
            vz = abs(self.vel[2])
            align = math.acos(float(np.clip(np.dot(axis, self.pad_normal),
                                            -1.0, 1.0)))
            slide = vxy ** 2 / (2 * self.mu * G0)
            on_pad = dist_xy <= self.PAD_R
            soft = vz <= self.MAX_VZ and vxy <= self.MAX_VXY
            aligned = align <= self.MAX_ALIGN
            stuck = dist_xy + slide <= self.PAD_R
            if on_pad and soft and aligned and stuck:
                outcome = "landed"
                reward += 120.0 - 4.0 * vz - 2.0 * vxy - 30.0 * align
            else:
                outcome = "crashed"
                reward -= 80.0
                fails = []
                if not on_pad:
                    fails.append("off-pad")
                if not soft:
                    fails.append("too-fast")
                if not aligned:
                    fails.append("bad-attitude")
                if on_pad and not stuck:
                    fails.append("slid-off")
                reason = "+".join(fails) if fails else "touchdown"
        elif abs(self.pos[0]) > 150 or abs(self.pos[1]) > 150 \
                or self.pos[2] > 200:
            terminated, outcome, reason = True, "crashed", "flew-away"
            reward -= 80.0
        elif self.steps >= self.MAX_STEPS:
            truncated, outcome = True, "timeout"
            reward -= 30.0

        info = {
            "outcome": outcome,
            "reason": reason,
            "fuel_used_frac": 1.0 - fuel_frac,
            "flight_time": self.t,
            "waypoint_reached": self.wp_reached,
            "phase": self.phase,
            "engine_id": self.engine_id,
            "tech_id": self.tech_id,
            "weather_id": self.weather_id,
        }
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ obs
    def _obs(self):
        lo = self.loadout
        wind = self.wind_base + self.gust
        return np.array([
            self.pos[0] / 60.0, self.pos[1] / 60.0, self.pos[2] / 80.0,
            self.vel[0] / 25.0, self.vel[1] / 25.0, self.vel[2] / 25.0,
            self.tilt[0] / 0.5, self.tilt[1] / 0.5,
            self.tilt_vel[0] / 2.0, self.tilt_vel[1] / 2.0,
            self.fuel / lo["fuel_mass"],
            wind[0] / 10.0, wind[1] / 10.0, wind[2] / 10.0,
            self.throttle,
            float(self.phase),
            self.waypoint[0] / 60.0, self.waypoint[1] / 60.0,
            self.waypoint[2] / 80.0,
            self.pad[0] / 60.0, self.pad[1] / 60.0,
            self.slope[0] / self.SLOPE_MAX, self.slope[1] / self.SLOPE_MAX,
            self.mu,
            WEATHER_IDS.index(self.weather_id) / 3.0,
            lo["thrust_sl"] / (lo["wet_mass"] * G0) / 2.0,
            lo["mdot_full"] / 900.0,
            lo["throttle_min"],
        ], dtype=np.float32)


if __name__ == "__main__":
    # smoke test: random loadouts, hover-ish policy
    import itertools
    combos = [("raptor3", "standard"), ("merlin1d", "rdre"),
              ("rutherford", "aerospike"), ("be4", "rdre_spike")]
    for eng, tech in combos:
        env = LoadoutMissionEnv(engine_id=eng, tech_id=tech, seed=7)
        obs, _ = env.reset()
        done, ret = False, 0.0
        while not done:
            a = np.array([0.3, 0.0, 0.0])  # 65% throttle-ish, no tilt
            obs, rew, term, trunc, info = env.step(a)
            ret += rew
            done = term or trunc
        print(f"{eng}+{tech}: outcome={info['outcome']} "
              f"reason={info.get('reason')} t={info['flight_time']:.1f}s "
              f"fuel_used={info['fuel_used_frac']:.2f} wp={info['waypoint_reached']}")
    print("obs dim:", env.observation_space.shape)
