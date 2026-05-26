"""Short video of any 2D LWR case from 2d_lwr_visualization.py.

Reuses the solver and IC builders from that module. Runs the same FV scheme
to t=1, records a frame every ``--frame-every`` timesteps, and writes an
MP4 (via matplotlib's FFMpegWriter) or a GIF (via PillowWriter) fallback.

Run locally::

    python 2d_lwr_circular_patch_video.py
    python 2d_lwr_circular_patch_video.py --case planar_shock
    python 2d_lwr_circular_patch_video.py --case oblique_shock --format gif

Output lands at ``outputs/2d_lwr_<case>.<mp4|gif>``.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from importlib import import_module

# Import the existing solver module so we stay in sync with the spec.
_viz = import_module("2d_lwr_visualization")
step = _viz.step
make_initial_condition = _viz.make_initial_condition
N = _viz.N
CFL = _viz.CFL
T = _viz.T
BETA = _viz.BETA
OUT_DIR = _viz.OUT_DIR


def run_and_collect_frames(case: str, frame_every: int):
    """Solve the chosen case to T, returning (times, frames, dx, dy)."""
    dx = 2.0 / N
    dy = 2.0 / N
    xs = np.linspace(-1.0 + 0.5 * dx, 1.0 - 0.5 * dx, N)
    ys = np.linspace(-1.0 + 0.5 * dy, 1.0 - 0.5 * dy, N)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    rho, _ = make_initial_condition(case, X, Y)

    dt = CFL / (1.0 / dx + BETA / dy)
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps                              # land exactly on T

    times = [0.0]
    frames = [rho.copy()]
    for n in range(1, n_steps + 1):
        rho = step(rho, dx, dy, dt, BETA)
        if n % frame_every == 0 or n == n_steps:
            times.append(n * dt)
            frames.append(rho.copy())

    print(f"steps={n_steps} dt={dt:.5g} frames={len(frames)}")
    return times, frames, dx, dy


def make_animation(case: str, times, frames, fps: int):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    im = ax.imshow(
        frames[0], origin="lower", extent=[-1, 1, -1, 1],
        vmin=0.0, vmax=1.0, interpolation="nearest", cmap="viridis",
    )
    title = ax.set_title(f"2D LWR {case} — t = {times[0]:.3f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\rho$")

    def update(i):
        im.set_data(frames[i])
        title.set_text(f"2D LWR {case} — t = {times[i]:.3f}")
        return im, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000.0 / fps, blit=False,
    )
    return fig, anim


def save_animation(anim, fig, out_base: str, fps: int, fmt: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    fmt = fmt.lower()

    if fmt == "mp4":
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
            out_path = out_base + ".mp4"
            anim.save(out_path, writer=writer, dpi=150)
            plt.close(fig)
            return out_path
        except (FileNotFoundError, RuntimeError) as e:
            print(f"FFmpeg unavailable ({e}); falling back to GIF.")
            fmt = "gif"

    if fmt == "gif":
        writer = animation.PillowWriter(fps=fps)
        out_path = out_base + ".gif"
        anim.save(out_path, writer=writer, dpi=120)
        plt.close(fig)
        return out_path

    raise ValueError(f"unknown format: {fmt}")


def main():
    p = argparse.ArgumentParser(
        description="Render a short video of the 2D LWR circular-patch case."
    )
    p.add_argument("--fps", type=int, default=24,
                   help="Frames per second in the output video (default 24).")
    p.add_argument("--frame-every", type=int, default=2,
                   help="Record one frame every N solver steps (default 2). "
                        "Lower = smoother + bigger file.")
    p.add_argument("--format", choices=("mp4", "gif"), default="mp4",
                   help="Output container. Falls back to GIF if FFmpeg "
                        "is missing.")
    p.add_argument("--hold-initial", type=float, default=1.0,
                   help="Seconds to hold on the initial condition before the "
                        "solver starts moving (default 1.0). Set to 0 to "
                        "disable.")
    p.add_argument("--case", type=str, default="circular_patch",
                   choices=("planar_shock", "planar_rarefaction",
                            "oblique_shock", "oblique_rarefaction",
                            "circular_patch", "quadrant_riemann",
                            "fast_oblique_shock"),
                   help="Which IC from 2d_lwr_visualization.py to animate "
                        "(default circular_patch).")
    args = p.parse_args()

    times, frames, _, _ = run_and_collect_frames(args.case, args.frame_every)

    # Repeat frame 0 so the viewer actually registers the initial condition
    # before the dynamics kick in.
    hold_frames = max(int(round(args.hold_initial * args.fps)), 0)
    if hold_frames > 0:
        times = [times[0]] * hold_frames + times
        frames = [frames[0]] * hold_frames + frames

    fig, anim = make_animation(args.case, times, frames, args.fps)
    out_base = os.path.join(OUT_DIR, f"2d_lwr_{args.case}")
    out_path = save_animation(anim, fig, out_base, args.fps, args.format)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
