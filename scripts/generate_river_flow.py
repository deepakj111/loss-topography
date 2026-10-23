#!/usr/bin/env python3
"""
Generates the flagship 'Optimization River Flow' animated GIF.

Simulates an ensemble of gradient-flow particles streaming downhill like a
liquid river across a complex non-convex landscape, showing how optimization
mechanics mirror hydrodynamic flow down potential energy gradients.
"""

from __future__ import annotations
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.gridspec as gridspec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loss_topography.theme import (
    apply_mpl_theme, BG_CANVAS, BG_PANEL, GRID_COLOR,
    TEXT_HEADER, TEXT_LABEL, TEXT_TICK, DARK_CMAP,
)
from loss_topography.surfaces import MultiModal
from loss_topography.optimizers import Adam, MomentumSGD

ASSETS_DIR = os.path.join(ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
apply_mpl_theme()


def generate_optimization_river_flow():
    print("  🌊 Generating optimization_river_flow.gif ...")

    surface = MultiModal()
    X, Y, Z = surface.meshgrid(n=250)
    Z_clip = np.clip(Z, np.percentile(Z, 1), np.percentile(Z, 98))

    # Initialize a swarm of particles forming headwaters along the ridge
    np.random.seed(42)
    n_particles = 28
    n_steps = 110

    # Start points along an arc on the high ground
    angles = np.linspace(0.1, 1.4, n_particles)
    r_base = 2.8 + 0.3 * np.random.randn(n_particles)
    start_xs = r_base * np.cos(angles)
    start_ys = r_base * np.sin(angles)

    # Color gradient for river tributaries: cyan to teal to emerald to violet
    river_palette = [
        "#38BDF8", "#0EA5E9", "#0284C7", "#06B6D4", "#14B8A6",
        "#2DD4BF", "#10B981", "#34D399", "#6EE7B7", "#818CF8",
        "#A78BFA", "#C084FC", "#38BDF8", "#0EA5E9", "#2DD4BF",
        "#34D399", "#38BDF8", "#22D3EE", "#06B6D4", "#0284C7",
        "#14B8A6", "#2DD4BF", "#10B981", "#6EE7B7", "#818CF8",
        "#A78BFA", "#38BDF8", "#2DD4BF",
    ]

    # Simulate trajectories using adaptive momentum with hydrodynamic viscosity
    trajectories = []
    for i in range(n_particles):
        x0, y0 = float(start_xs[i]), float(start_ys[i])
        # Adam optimizer with varied learning rates for realistic laminar/turbulent flow
        lr = 0.05 + 0.02 * (i % 5) / 5.0
        opt = Adam(lr=lr, beta1=0.88, beta2=0.98)
        log = opt.run(surface, x0, y0, n_steps=n_steps)
        trajectories.append(log)

    # Set up figure
    fig = plt.figure(figsize=(13, 5.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 0.9], wspace=0.08)

    # Left: 3D Surface with flowing river ribbons
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax3d.set_facecolor(BG_CANVAS)
    surf = ax3d.plot_surface(
        X, Y, Z_clip, cmap=DARK_CMAP, alpha=0.55,
        rstride=3, cstride=3, linewidth=0, antialiased=True,
    )
    ax3d.set_xlabel("θ₁", color=TEXT_LABEL, fontsize=9)
    ax3d.set_ylabel("θ₂", color=TEXT_LABEL, fontsize=9)
    ax3d.set_zlabel("Loss", color=TEXT_LABEL, fontsize=9)
    ax3d.set_title("3D Loss Topography — Hydrodynamic Stream Flow",
                   color=TEXT_HEADER, fontsize=11, fontweight="bold", pad=12)
    ax3d.tick_params(colors=TEXT_TICK, labelsize=7)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor(GRID_COLOR)
    ax3d.yaxis.pane.set_edgecolor(GRID_COLOR)
    ax3d.zaxis.pane.set_edgecolor(GRID_COLOR)

    # Right: 2D Contour Map
    ax2d = fig.add_subplot(gs[1])
    ax2d.set_facecolor(BG_PANEL)
    levels = np.linspace(Z_clip.min(), Z_clip.max(), 35)
    ax2d.contourf(X, Y, Z_clip, levels=levels, cmap=DARK_CMAP, alpha=0.65)
    ax2d.contour(X, Y, Z_clip, levels=levels[::4], colors=[GRID_COLOR],
                 linewidths=0.4, alpha=0.4)
    ax2d.set_xlabel("Parameter θ₁", color=TEXT_LABEL, fontsize=9)
    ax2d.set_ylabel("Parameter θ₂", color=TEXT_LABEL, fontsize=9)
    ax2d.set_title("Optimization River Bed (2D Contour View)",
                   color=TEXT_HEADER, fontsize=11, fontweight="bold")
    ax2d.tick_params(colors=TEXT_TICK, labelsize=8)
    ax2d.set_aspect("equal")

    # Animated elements
    lines_3d = []
    heads_3d = []
    lines_2d = []
    heads_2d = []

    for i in range(n_particles):
        col = river_palette[i % len(river_palette)]
        l3d, = ax3d.plot([], [], [], color=col, lw=1.6, alpha=0.85)
        h3d, = ax3d.plot([], [], [], 'o', color="#FFFFFF", markeredgecolor=col,
                         markersize=4.5, markeredgewidth=1.2, zorder=15)
        l2d, = ax2d.plot([], [], color=col, lw=1.8, alpha=0.85)
        h2d, = ax2d.plot([], [], 'o', color="#FFFFFF", markeredgecolor=col,
                         markersize=5, markeredgewidth=1.2, zorder=15)

        lines_3d.append(l3d)
        heads_3d.append(h3d)
        lines_2d.append(l2d)
        heads_2d.append(h2d)

    # Telemetry HUD
    hud = ax2d.text(
        0.03, 0.03, "", transform=ax2d.transAxes,
        fontsize=7.5, family="monospace", color=TEXT_HEADER,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_PANEL,
                  edgecolor=GRID_COLOR, alpha=0.92),
    )

    # Subsample frames for lean GIF size
    frame_indices = list(range(0, n_steps + 1, 2))

    def animate(frame_idx):
        step = frame_indices[frame_idx]
        cur_losses = []

        # Smooth camera rotation
        azim = -55.0 + 35.0 * (frame_idx / len(frame_indices))
        ax3d.view_init(elev=34, azim=azim)

        for i in range(n_particles):
            log = trajectories[i]
            cur_idx = min(step, len(log.xs) - 1)
            # Flow trail: full history up to cur_idx
            xs = log.xs[:cur_idx + 1]
            ys = log.ys[:cur_idx + 1]
            zs = log.losses[:cur_idx + 1]

            cur_losses.append(zs[-1])

            lines_3d[i].set_data(xs, ys)
            lines_3d[i].set_3d_properties(zs)
            heads_3d[i].set_data([xs[-1]], [ys[-1]])
            heads_3d[i].set_3d_properties([zs[-1]])

            lines_2d[i].set_data(xs, ys)
            heads_2d[i].set_data([xs[-1]], [ys[-1]])

        mean_l = np.mean(cur_losses)
        min_l = np.min(cur_losses)
        flow_pct = min(100.0, (step / n_steps) * 100)

        hud.set_text(
            f"STREAM TELEMETRY (t={step:3d})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Tributaries : {n_particles} streams\n"
            f"Mean Loss   : {mean_l:7.3f}\n"
            f"Deepest Loss: {min_l:7.3f}\n"
            f"Basin Depth : {flow_pct:5.1f}% descent"
        )
        return []

    anim = FuncAnimation(
        fig, animate, frames=len(frame_indices),
        interval=50, blit=False,
    )
    path = os.path.join(ASSETS_DIR, "optimization_river_flow.gif")
    anim.save(path, writer=PillowWriter(fps=18), dpi=85)
    plt.close(fig)
    print(f"    ✓ Saved {path} ({os.path.getsize(path) / 1024:.0f} KB)")


if __name__ == "__main__":
    generate_optimization_river_flow()
