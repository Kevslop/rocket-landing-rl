"""
env.py — Custom 2D rocket landing environment (Gymnasium API).

Original code written for this project. Physics is a simple 2D rigid-body
simulation: point gravity, a main thruster along the rocket's axis, two side
thrusters for attitude control, linear drag, and a flat world with a landing
pad. No code is copied from any existing rocket-landing repo.

State (8 floats, Box):
    [x, y, vx, vy, angle, angular_velocity, leg_contact_left, leg_contact_right]
    - x, y: position in world units (x in [-1, 1], y in [0, 1.2]; pad at x=0)
    - vx, vy: velocity
    - angle: tilt from vertical, radians, positive = leaning right
    - angular_velocity: rad/s
    - leg contacts: 1.0 if that leg is touching the ground, else 0.0

Actions (MultiDiscrete, 3 sub-actions):
    - main engine:  0 = off, 1 = half thrust, 2 = full thrust
    - left RCS:     0 = off, 1 = fire  (torque, rotates rocket left)
    - right RCS:    0 = off, 1 = fire  (torque, rotates rocket right)
Discrete thrust levels keep the action space small enough for PPO to learn
quickly while still allowing fine control.

Fuel: the main engine and RCS draw from one tank. Firing with an empty tank
does nothing. Fuel is part of the observation indirectly (see fuel_left
property) — the observation itself stays 8-D to keep the policy simple, and
the shaped reward penalises fuel burn so the agent learns to be thrifty.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - gymnasium is a hard dependency
    raise ImportError("gymnasium is required: pip install -r requirements.txt")


class RocketLandingEnv(gym.Env):
    """2D rocket landing task. Land softly, upright, on the pad."""

    metadata = {"render_modes": ["rgb_array"]}

    # ---- world constants -------------------------------------------------
    GRAVITY = 9.81          # m/s^2 downward
    WORLD_W = 2.0           # world spans x in [-1, 1]
    WORLD_H = 1.2           # world spans y in [0, 1.2]
    PAD_HALF_WIDTH = 0.18   # landing pad spans x in [-0.18, 0.18]
    PAD_Y = 0.0             # pad sits on the ground (y = 0)
    MAX_STEPS = 600         # episode timeout (10 s at dt = 1/60)

    # ---- rocket constants ------------------------------------------------
    MASS = 1.0              # kg (thrust is expressed as acceleration)
    MAIN_THRUST = 22.0      # m/s^2 at full throttle (> gravity, so it can hover/climb)
    RCS_TORQUE = 6.0        # rad/s^2 from one side thruster
    DRAG = 0.10             # linear drag coefficient (1/s)
    ANGULAR_DRAG = 0.30     # angular damping (1/s)
    FUEL_MAX = 100.0        # fuel units
    MAIN_FUEL_RATE = 0.18   # fuel / step at full throttle (half throttle = half)
    RCS_FUEL_RATE = 0.03    # fuel / step per RCS thruster
    ROCKET_HALF_HEIGHT = 0.06  # half of the rocket body height (for ground contact)

    # ---- landing criteria ------------------------------------------------
    MAX_LANDING_VY = 1.5    # |vertical speed| allowed for a soft landing
    MAX_LANDING_VX = 1.0    # |horizontal speed| allowed
    MAX_LANDING_ANGLE = 0.20  # |tilt| allowed (radians, ~11 degrees)

    def __init__(self, wind: float = 0.0, seed: int | None = None,
                 render_mode: str | None = None):
        super().__init__()
        self.wind = wind  # constant horizontal wind acceleration (m/s^2)
        self.render_mode = render_mode

        # Observation: [x, y, vx, vy, angle, ang_vel, leg_L, leg_R]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, -5.0, -5.0, -math.pi, -5.0, 0.0, 0.0],
                         dtype=np.float32),
            high=np.array([1.0, 1.2, 5.0, 5.0, math.pi, 5.0, 1.0, 1.0],
                          dtype=np.float32),
            dtype=np.float32,
        )
        # Action: [main (0/1/2), left RCS (0/1), right RCS (0/1)]
        self.action_space = spaces.MultiDiscrete([3, 2, 2])

        self._rng = np.random.default_rng(seed)
        self.state: np.ndarray | None = None
        self.fuel = self.FUEL_MAX
        self.steps = 0
        self.trajectory: list[tuple[float, float]] = []
        self.landed = False
        self.crashed = False

    # ---- Gymnasium API ---------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        # Start above the pad with a randomised offset / velocity / tilt so the
        # policy must actually fly, not memorise one trajectory.
        x = float(rng.uniform(-0.45, 0.45))
        y = float(rng.uniform(0.85, 1.10))
        vx = float(rng.uniform(-0.3, 0.3))
        vy = float(rng.uniform(-0.5, 0.0))
        angle = float(rng.uniform(-0.25, 0.25))
        ang_vel = float(rng.uniform(-0.2, 0.2))
        self.state = np.array([x, y, vx, vy, angle, ang_vel, 0.0, 0.0],
                              dtype=np.float32)
        self.fuel = self.FUEL_MAX
        self.steps = 0
        self.trajectory = [(x, y)]
        self.landed = False
        self.crashed = False
        return self._obs(), {}

    def step(self, action):
        assert self.state is not None, "call reset() before step()"
        main, left_rcs, right_rcs = (int(a) for a in np.asarray(action).flatten())
        dt = 1.0 / 60.0
        x, y, vx, vy, angle, ang_vel, _, _ = (float(v) for v in self.state)

        # --- thrust (fuel-limited) ---
        throttle = [0.0, 0.5, 1.0][main]
        if self.fuel <= 0:
            throttle = 0.0
            left_rcs = right_rcs = 0
        # Main engine thrust points along the rocket's axis (angle from vertical).
        ax = math.sin(angle) * throttle * self.MAIN_THRUST + self.wind
        ay = math.cos(angle) * throttle * self.MAIN_THRUST - self.GRAVITY
        # Side thrusters: torque only (no net translation).
        torque = (right_rcs - left_rcs) * self.RCS_TORQUE
        # Fuel burn.
        self.fuel = max(0.0, self.fuel
                         - throttle * self.MAIN_FUEL_RATE
                         - (left_rcs + right_rcs) * self.RCS_FUEL_RATE)

        # --- integrate (semi-implicit Euler) with drag ---
        vx = vx + ax * dt - self.DRAG * vx * dt
        vy = vy + ay * dt - self.DRAG * vy * dt
        ang_vel = ang_vel + torque * dt - self.ANGULAR_DRAG * ang_vel * dt
        angle = angle + ang_vel * dt
        angle = (angle + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
        x = x + vx * dt
        y = y + vy * dt
        self.steps += 1

        # --- ground contact ---
        # The rocket rests on its legs when its base reaches the ground.
        # Leg contact is computed from the body's bottom corners.
        leg_l = leg_r = 0.0
        ground_hit = y - self.ROCKET_HALF_HEIGHT <= 0.0
        if ground_hit:
            y = self.ROCKET_HALF_HEIGHT
            # Legs touch if the body is roughly upright.
            if abs(angle) < 0.6:
                leg_l = leg_r = 1.0
            # Resting contact kills velocity (inelastic ground).
            vx, vy, ang_vel = 0.0, 0.0, 0.0

        self.state = np.array([x, y, vx, vy, angle, ang_vel, leg_l, leg_r],
                              dtype=np.float32)
        self.trajectory.append((x, y))

        # --- termination ---
        terminated = False
        reward = self._shaped_reward(x, y, vx, vy, angle, throttle,
                                     left_rcs, right_rcs)
        if ground_hit:
            on_pad = abs(x) <= self.PAD_HALF_WIDTH
            soft = abs(vy) <= self.MAX_LANDING_VY
            slow = abs(vx) <= self.MAX_LANDING_VX
            upright = abs(angle) <= self.MAX_LANDING_ANGLE
            if on_pad and soft and slow and upright:
                terminated = True
                self.landed = True
                reward += 100.0  # big bonus for a clean landing
            else:
                terminated = True
                self.crashed = True
                # Crash penalty scales with how bad the touchdown was.
                reward -= 50.0 + 10.0 * min(abs(vy), 5.0) + 20.0 * abs(angle)
        elif x < -1.0 or x > 1.0 or y > self.WORLD_H:
            terminated = True
            self.crashed = True
            reward -= 100.0  # flew off-screen
        elif self.steps >= self.MAX_STEPS:
            terminated = False  # timeout = truncation, not failure
            reward -= 10.0

        truncated = (self.steps >= self.MAX_STEPS) and not terminated
        info = {
            "landed": self.landed,
            "crashed": self.crashed,
            "fuel_left": self.fuel,
            "steps": self.steps,
        }
        return self._obs(), float(reward), terminated, truncated, info

    # ---- helpers ---------------------------------------------------------
    def _obs(self) -> np.ndarray:
        return np.array(self.state, dtype=np.float32)

    def _shaped_reward(self, x, y, vx, vy, angle, throttle, left_rcs, right_rcs):
        """Dense shaping: pull the rocket toward a gentle, upright descent
        onto the pad, and penalise fuel burn + tilt so the policy stays
        efficient and stable instead of hovering forever."""
        # Distance to pad centre (3D-ish: horizontal + altitude).
        dist = math.hypot(x, y - self.PAD_Y)
        reward = -0.30 * dist
        # Penalise speed (want a slow, controlled descent).
        reward -= 0.10 * (abs(vx) + abs(vy))
        # Penalise tilt (want near-vertical).
        reward -= 0.20 * abs(angle)
        # Small per-step penalty to encourage finishing promptly.
        reward -= 0.01
        # Fuel penalty: full throttle costs more than half throttle.
        reward -= 0.02 * throttle + 0.005 * (left_rcs + right_rcs)
        # Bonus for being low, slow, and over the pad (approach shaping).
        if abs(x) < self.PAD_HALF_WIDTH and 0.0 < y < 0.4:
            reward += 0.10 * (1.0 - abs(vy) / 5.0)
        return reward

    @property
    def fuel_left(self) -> float:
        return self.fuel


if __name__ == "__main__":
    # Quick sanity check: random policy should terminate every episode.
    env = RocketLandingEnv()
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        done = False
        total = 0.0
        while not done:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            total += r
            done = term or trunc
        print(f"episode: steps={info['steps']} landed={info['landed']} "
              f"reward={total:.1f} fuel_left={info['fuel_left']:.1f}")
        obs, _ = env.reset()
