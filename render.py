"""
render.py — Matplotlib rendering for the rocket landing environment.

Headless-safe: every function draws with the Agg backend and returns
numpy RGB arrays (for GIFs) or saves PNG/GIF files directly. No window is
ever opened, so this works on servers, CI, and SSH sessions.

The drawing shows:
  - the ground line and the landing pad (green zone)
  - the rocket body (rotated rectangle), legs, and engine flame
  - the trajectory trace (fading dots)
  - live telemetry text (altitude, velocity, angle, fuel)
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Colours (kept muted so the GIF stays readable).
SKY = "#0b1026"
PAD = "#2ecc71"
PAD_EDGE = "#27ae60"
ROCKET_BODY = "#ecf0f1"
ROCKET_NOSE = "#e74c3c"
FLAME = "#f39c12"
TRACE = "#3498db"
TEXT = "#ecf0f1"


def draw_frame(env, ax, show_trace: bool = True):
    """Draw one frame of `env` onto matplotlib Axes `ax`."""
    ax.set_facecolor(SKY)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.12, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # Ground + pad.
    ax.axhline(0, color="#7f8c8d", lw=2)
    pad = Rectangle((-env.PAD_HALF_WIDTH, -0.03),
                    2 * env.PAD_HALF_WIDTH, 0.03,
                    facecolor=PAD, edgecolor=PAD_EDGE, lw=1.5, zorder=2)
    ax.add_patch(pad)
    ax.text(0, -0.10, "LANDING PAD", ha="center", va="top",
            color=PAD, fontsize=7, weight="bold")

    # Trajectory trace.
    if show_trace and len(env.trajectory) > 1:
        xs, ys = zip(*env.trajectory)
        ax.plot(xs, ys, color=TRACE, lw=1.0, alpha=0.55, zorder=1)

    # Rocket: body is a rotated rectangle centred on (x, y).
    x, y, vx, vy, angle, ang_vel, leg_l, leg_r = (float(v) for v in env.state)
    w, h = 0.045, 0.12  # body width / height
    body = Rectangle((x - w / 2, y - h / 2), w, h,
                     facecolor=ROCKET_BODY, edgecolor="#2c3e50", lw=1.2,
                     zorder=4)
    # Rotate around the rocket centre.
    t = matplotlib.transforms.Affine2D().rotate_around(x, y, -angle) + ax.transData
    body.set_transform(t)
    ax.add_patch(body)
    # Nose cone (small triangle at the top, same rotation).
    nose_x = [x - w / 2, x + w / 2, x]
    nose_y = [y + h / 2, y + h / 2, y + h / 2 + 0.03]
    nose = plt.Polygon(list(zip(nose_x, nose_y)), closed=True,
                       facecolor=ROCKET_NOSE, edgecolor="#2c3e50", lw=0.8,
                       zorder=4)
    nose.set_transform(t)
    ax.add_patch(nose)
    # Legs (two short lines splayed from the base).
    for side in (-1, 1):
        lx = [x + side * w / 2, x + side * 0.055]
        ly = [y - h / 2, y - h / 2 - 0.035]
        leg = plt.Line2D(lx, ly, color="#bdc3c7", lw=1.6, zorder=3)
        leg.set_transform(t)
        ax.add_line(leg)

    # Engine flame: drawn when the main engine fired on the last step.
    # We infer it from the action history stored on the env by evaluate.py
    # (env.last_action) — absent in plain env use, so default to no flame.
    last = getattr(env, "last_action", None)
    if last is not None and int(last[0]) > 0:
        throttle = [0.0, 0.5, 1.0][int(last[0])]
        flame_len = 0.05 + 0.08 * throttle
        # Flame points opposite the rocket axis (downward when upright).
        fx = x - math.sin(angle) * (h / 2 + flame_len / 2)
        fy = y - math.cos(angle) * (h / 2 + flame_len / 2)
        flame = plt.Polygon(
            [(x - math.sin(angle) * h / 2 - math.cos(angle) * w / 3,
              y - math.cos(angle) * h / 2 + math.sin(angle) * w / 3),
             (x - math.sin(angle) * h / 2 + math.cos(angle) * w / 3,
              y - math.cos(angle) * h / 2 - math.sin(angle) * w / 3),
             (fx, fy)],
            closed=True, facecolor=FLAME, edgecolor="#e67e22",
            lw=0.6, alpha=0.9, zorder=3)
        ax.add_patch(flame)

    # Telemetry.
    speed = math.hypot(vx, vy)
    ax.text(-1.02, 1.20, f"alt {max(y, 0):.2f} m", color=TEXT, fontsize=8,
            va="top", family="monospace")
    ax.text(-1.02, 1.12, f"vel {speed:.2f} m/s", color=TEXT, fontsize=8,
            va="top", family="monospace")
    ax.text(-1.02, 1.04, f"tilt {math.degrees(angle):+.1f} deg", color=TEXT,
            fontsize=8, va="top", family="monospace")
    ax.text(0.45, 1.20, f"fuel {env.fuel:.0f}", color=TEXT, fontsize=8,
            va="top", ha="left", family="monospace")
    ax.text(0.45, 1.12, f"step {env.steps}", color=TEXT, fontsize=8,
            va="top", ha="left", family="monospace")


def frame_array(env, figsize=(6.4, 6.4), dpi=100) -> np.ndarray:
    """Render one frame to an HxWx3 uint8 RGB array."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    try:
        draw_frame(env, ax)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()
    finally:
        plt.close(fig)


def save_gif(frames: list[np.ndarray], path: str, fps: int = 30):
    """Save a list of RGB frames as a GIF (matplotlib animation writer)."""
    from matplotlib import animation

    if not frames:
        raise ValueError("no frames to save")
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    ax.axis("off")
    img = ax.imshow(frames[0])
    ax.set_xlim(0, frames[0].shape[1])
    ax.set_ylim(frames[0].shape[0], 0)

    def update(i):
        img.set_data(frames[i])
        return [img]

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 / fps, blit=True)
    try:
        anim.save(path, writer="pillow", fps=fps)
    finally:
        plt.close(fig)
    return path
