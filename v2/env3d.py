"""
env3d.py — 3D rocket landing environment (Gymnasium API) with domain randomization.

Original code written for this project. The vehicle is a 3D point mass with
tilt (gimbal) dynamics: the main engine gimbals within a limited angle and
the thrust vector follows the commanded tilt with first-order lag. Gravity,
linear aerodynamic drag, time-varying wind gusts, and fuel burn (which
lightens the vehicle) are simulated. The landing pad is a disc at the
origin; touchdown is a success only if the rocket is on the pad, slow, and
near-upright.

Every episode randomizes (domain randomization for robust training):
    - initial position (x, y, altitude)
    - initial velocity
    - initial tilt and tilt rate
    - wind: base vector + Ornstein-Uhlenbeck gusts
    - dry mass (+/-10%) and fuel load (70-100%)

Observation (15 floats, all roughly normalized):
    [x, y, z, vx, vy, vz, tilt_x, tilt_z, tilt_vx, tilt_vz,
     fuel_frac, wind_x, wind_y, wind_z, throttle_prev]

Action (Box(3) in [-1, 1]):
    [throttle_cmd, tilt_x_cmd, tilt_z_cmd]
    throttle 0..1 mapped from [-1, 1]; tilt commands in radians (gimbal limit).
"""

from __future__ import annotations

import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    raise ImportError("gymnasium is required: pip install -r v2/requirements.txt")


class RocketLanding3DEnv(gym.Env):
    """3D rocket landing task with gimballed thrust and wind gusts."""

    metadata = {"render_modes": ["rgb_array"]}

    # ---- simulation ------------------------------------------------------
    DT = 0.05               # physics step (20 Hz)
    MAX_STEPS = 900         # episode timeout (45 s)
    GRAVITY = 9.81

    # ---- vehicle ---------------------------------------------------------
    DRY_MASS = 8000.0       # kg, randomized +/-10%
    FUEL_MASS = 3000.0      # kg at 100% load
    FULL_BURN_TIME = 30.0   # seconds of full throttle per full tank
    MAX_THRUST_TW = 1.7     # max thrust / (wet weight) at full fuel
    GIMBAL_LIMIT = 0.35     # rad (~20 deg) max tilt command
    TILT_K = 6.0            # tilt servo stiffness (1/s^2)
    TILT_D = 4.5            # tilt servo damping (1/s)
    DRAG_K = 1400.0         # linear drag coefficient (N per m/s of airspeed)

    # ---- world -----------------------------------------------------------
    PAD_RADIUS = 5.0        # m, landing pad disc radius
    START_XY = 30.0         # initial |x|,|y| up to this (m)
    START_Z = (35.0, 60.0)  # initial altitude range (m)
    START_VXY = 8.0         # initial |vx|,|vy| up to this (m/s)
    START_VZ = (-6.0, 2.0)  # initial vertical speed range (m/s)
    START_TILT = 0.15       # initial tilt up to this (rad)
    WIND_BASE = 6.0         # base wind speed up to this (m/s)
    WIND_GUST = 4.0         # gust OU-process sigma scale (m/s)
    WIND_TAU = 2.5          # gust correlation time (s)

    # ---- landing criteria ------------------------------------------------
    MAX_LANDING_VZ = 7.0    # |vertical speed| allowed (m/s)
    MAX_LANDING_VXY = 5.0   # |horizontal speed| allowed (m/s)
    MAX_LANDING_TILT = 0.30  # |tilt| allowed (rad)

    def __init__(self, seed: int | None = None, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # 15-D normalized observation (see module docstring)
        obs_low = np.array(
            [-2, -2, -0.2, -2, -2, -2, -2, -2, -3, -3, 0, -2, -2, -2, 0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 1, 2, 2, 2, 1], dtype=np.float32
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ------------------------------------------------------------------ api
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        r = self._rng
        # --- domain randomization -------------------------------------
        self.dry_mass = self.DRY_MASS * r.uniform(0.9, 1.1)
        self.fuel = self.FUEL_MASS * r.uniform(0.7, 1.0)
        self.fuel_full = self.FUEL_MASS
        self.max_thrust = (
            self.MAX_THRUST_TW * (self.dry_mass + self.fuel_full) * self.GRAVITY
        )
        self.fuel_rate = self.fuel_full / self.FULL_BURN_TIME  # kg/s at full

        self.pos = np.array(
            [
                r.uniform(-self.START_XY, self.START_XY),
                r.uniform(-self.START_XY, self.START_XY),
                r.uniform(*self.START_Z),
            ],
            dtype=np.float64,
        )
        self.vel = np.array(
            [
                r.uniform(-self.START_VXY, self.START_VXY),
                r.uniform(-self.START_VXY, self.START_VXY),
                r.uniform(*self.START_VZ),
            ],
            dtype=np.float64,
        )
        ang = r.uniform(0, 2 * math.pi)
        mag = r.uniform(0, self.START_TILT)
        self.tilt = np.array([mag * math.cos(ang), mag * math.sin(ang)])
        self.tilt_vel = r.uniform(-0.2, 0.2, size=2)
        self.throttle = 0.0

        # wind: random base direction/speed + OU gust state
        w_ang = r.uniform(0, 2 * math.pi)
        w_spd = r.uniform(0, self.WIND_BASE)
        self.wind_base = np.array(
            [w_spd * math.cos(w_ang), w_spd * math.sin(w_ang), 0.0]
        )
        self.gust = np.zeros(3)
        self.gust_sigma = r.uniform(0.3, 1.0) * self.WIND_GUST

        self.t = 0.0
        self.steps = 0
        self._prev_shaping = None
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        throttle_cmd = float((action[0] + 1.0) / 2.0)  # [-1,1] -> [0,1]
        tilt_cmd = np.clip(action[1:], -1.0, 1.0) * self.GIMBAL_LIMIT

        dt = self.DT
        r = self._rng

        # --- wind gusts (Ornstein-Uhlenbeck) ---------------------------
        theta = dt / self.WIND_TAU
        self.gust += -theta * self.gust + self.gust_sigma * math.sqrt(
            theta
        ) * r.normal(size=3)
        self.gust[2] *= 0.3  # vertical gusts are weaker
        wind = self.wind_base + self.gust

        # --- propulsion -------------------------------------------------
        fuel_frac_before = self.fuel / self.fuel_full if self.fuel_full else 0.0
        if self.fuel <= 0.0:
            self.throttle = 0.0
        else:
            # slight servo lag on throttle for realism
            self.throttle += np.clip(throttle_cmd - self.throttle, -dt * 3, dt * 3)
            self.throttle = float(np.clip(self.throttle, 0.0, 1.0))
            burn = self.throttle * self.fuel_rate * dt
            self.fuel = max(0.0, self.fuel - burn)

        mass = self.dry_mass + self.fuel

        # --- tilt servo --------------------------------------------------
        self.tilt_vel += (
            self.TILT_K * (tilt_cmd - self.tilt) - self.TILT_D * self.tilt_vel
        ) * dt
        self.tilt += self.tilt_vel * dt

        # thrust direction from tilt (z-up world: tilt[0] tips toward +x,
        # tilt[1] tips toward +y)
        tx, ty = self.tilt
        thrust_dir = np.array(
            [math.sin(tx), math.sin(ty), math.cos(tx) * math.cos(ty)]
        )
        thrust_dir /= np.linalg.norm(thrust_dir)

        # --- forces ------------------------------------------------------
        thrust = thrust_dir * (self.throttle * self.max_thrust)
        gravity = np.array([0.0, 0.0, -mass * self.GRAVITY])
        drag = self.DRAG_K * (wind - self.vel)  # N, opposes airspeed
        acc = (thrust + gravity + drag) / mass

        self.vel += acc * dt
        self.pos += self.vel * dt
        self.t += dt
        self.steps += 1

        # --- reward shaping ----------------------------------------------
        dist_xy = math.hypot(self.pos[0], self.pos[1])
        speed = float(np.linalg.norm(self.vel))
        tilt_mag = float(np.linalg.norm(self.tilt))
        shaping = -(dist_xy / 40.0 + self.pos[2] / 60.0 + speed / 25.0)
        if self.pos[2] < 12.0:  # stay upright when close to the ground
            shaping -= tilt_mag * 1.5
        reward = -0.002 - 0.01 * self.throttle  # time + fuel pressure
        if self._prev_shaping is not None:
            reward += (shaping - self._prev_shaping) * 2.0
        self._prev_shaping = shaping

        # --- termination --------------------------------------------------
        terminated = False
        outcome = None
        info = {"fuel_frac": self.fuel / self.fuel_full if self.fuel_full else 0.0}

        if self.pos[2] <= 0.0:
            self.pos[2] = 0.0
            terminated = True
            truncated = False
            vxy = math.hypot(self.vel[0], self.vel[1])
            vz = abs(self.vel[2])
            on_pad = dist_xy <= self.PAD_RADIUS
            soft = vz <= self.MAX_LANDING_VZ and vxy <= self.MAX_LANDING_VXY
            upright = tilt_mag <= self.MAX_LANDING_TILT
            if on_pad and soft and upright:
                outcome = "landed"
                reward += 100.0 - 4.0 * vz - 2.0 * vxy - 25.0 * tilt_mag
            else:
                outcome = "crashed"
                reward -= 60.0
            info.update(
                {"vz": vz, "vxy": vxy, "tilt": tilt_mag, "dist_xy": dist_xy}
            )
        elif abs(self.pos[0]) > 120 or abs(self.pos[1]) > 120 or self.pos[2] > 150:
            terminated = True
            truncated = False
            outcome = "crashed"  # flew away
            reward -= 60.0
        elif self.steps >= self.MAX_STEPS:
            terminated = False
            truncated = True
            outcome = "timeout"
            reward -= 20.0
        else:
            terminated = False
            truncated = False

        info["outcome"] = outcome
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ obs
    def _obs(self):
        wind = self.wind_base + self.gust
        return np.array(
            [
                self.pos[0] / 40.0,
                self.pos[1] / 40.0,
                self.pos[2] / 60.0,
                self.vel[0] / 20.0,
                self.vel[1] / 20.0,
                self.vel[2] / 20.0,
                self.tilt[0] / 0.5,
                self.tilt[1] / 0.5,
                self.tilt_vel[0] / 2.0,
                self.tilt_vel[1] / 2.0,
                self.fuel / self.fuel_full if self.fuel_full else 0.0,
                wind[0] / 10.0,
                wind[1] / 10.0,
                wind[2] / 10.0,
                self.throttle,
            ],
            dtype=np.float32,
        )


# ------------------------------------------------------------------ smoke test
if __name__ == "__main__":
    env = RocketLanding3DEnv(seed=0)
    for ep in range(3):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            # hover-ish random policy sanity check
            act = np.array([0.2, 0.0, 0.0])
            obs, rew, term, trunc, info = env.step(act)
            ret += rew
            done = term or trunc
        print(f"ep {ep}: outcome={info['outcome']} return={ret:.1f}")
    print("obs_space:", env.observation_space.shape, "act_space:", env.action_space.shape)
