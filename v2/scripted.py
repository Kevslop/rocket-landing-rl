"""
scripted.py — A simple PD controller that flies the mission. Used to:
1. Verify the mission is feasible.
2. Generate demonstrations for behavioral cloning.

The controller:
- Phase 0: climb to waypoint altitude while steering toward waypoint XY,
            slow down near waypoint, capture.
- Phase 1: fly toward pad, descend, flare, touchdown.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from mission_env import LoadoutMissionEnv


def scripted_action(env: LoadoutMissionEnv):
    """Returns action in [-1, 1]^3."""
    lo = env.loadout

    if env.phase == 0:
        target = env.waypoint
    else:
        target = np.array([env.pad[0], env.pad[1], 0.0])

    # --- horizontal PD: tilt toward target ---
    err_xy = target[:2] - env.pos[:2]
    # desired horizontal velocity: proportional, capped
    des_vxy = np.clip(err_xy * 0.8, -8, 8)
    # slow down when close
    dist_xy = float(np.linalg.norm(err_xy))
    if dist_xy < 10:
        des_vxy *= dist_xy / 10.0
    # tilt command: PD on velocity error
    # tilt angle ~ atan(accel/g); accel = kp*(des_v - v)
    kp = 1.5
    acc_cmd = kp * (des_vxy - env.vel[:2])
    # convert accel to tilt: tilt = atan(acc_xy / g)
    tilt_des = np.arctan2(np.linalg.norm(acc_cmd), 9.81)
    tilt_des = min(tilt_des, lo["gimbal_rad"] * 0.9)
    if np.linalg.norm(acc_cmd) > 0.1:
        dir_xy = acc_cmd / np.linalg.norm(acc_cmd)
        tilt_x = tilt_des * dir_xy[0]
        tilt_y = tilt_des * dir_xy[1]
    else:
        tilt_x, tilt_y = 0.0, 0.0

    # --- vertical: velocity command ---
    if env.phase == 0:
        # climb to waypoint, slow near it
        dz = target[2] - env.pos[2]
        des_vz = np.clip(dz * 0.8, -6, 8)
        if abs(dz) < 5:
            des_vz = np.clip(dz * 1.0, -3, 3)
    else:
        # descend toward pad, flare near ground
        alt = env.pos[2]
        if alt > 15:
            des_vz = -6.0
        elif alt > 5:
            des_vz = -3.0
        else:
            des_vz = -1.5  # gentle touchdown

    # throttle to track des_vz: P controller + hover feedforward
    # hover throttle ~= 1/1.35 = 0.74; map via our action mapping
    # thr = 0.85 + 0.35*a0  =>  a0 = (thr - 0.85)/0.35
    kv = 0.15
    thr_des = 0.74 + kv * (des_vz - env.vel[2])
    thr_des = float(np.clip(thr_des, lo["throttle_min"], 1.0))
    a0 = (thr_des - 0.85) / 0.35
    a0 = float(np.clip(a0, -1, 1))

    a1 = float(np.clip(tilt_x / lo["gimbal_rad"], -1, 1))
    a2 = float(np.clip(tilt_y / lo["gimbal_rad"], -1, 1))
    return np.array([a0, a1, a2])


def test(n=10, spawn="ground", pad_r=7.0):
    from collections import Counter
    c = Counter()
    for ep in range(n):
        env = LoadoutMissionEnv(engine_id="raptor3", tech_id="standard",
                                weather="clear", seed=50000 + ep,
                                spawn=spawn, pad_r=pad_r)
        obs, _ = env.reset()
        done = False
        while not done:
            a = scripted_action(env)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        key = info["outcome"] + (f"({info.get('reason')})"
                                 if info.get("reason") else "")
        c[key] += 1
    print(dict(c))
    return c


if __name__ == "__main__":
    print("ground, pad 7m:")
    test(10, "ground", 7.0)
    print("landing, pad 7m:")
    test(10, "landing", 7.0)
