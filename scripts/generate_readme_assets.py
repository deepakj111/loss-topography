#!/usr/bin/env python3
"""
Headless renderer that generates all GIFs and PNGs into assets/.

Generates:
  1. assets/hero_optimizer_race.gif
  2. assets/saddle_point_escape.gif
  3. assets/sharp_vs_flat_minima.gif
  4. assets/loss_function_atlas.png
  5. assets/loss_convergence_timing.png
  6. assets/filter_normalized_nn_surface.png
"""

from __future__ import annotations
import os
import sys
import numpy as np

# Ensure headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm, Normalize
import matplotlib.gridspec as gridspec

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loss_topography.theme import (
    apply_mpl_theme, OPTIMIZER_COLORS, BG_CANVAS, BG_PANEL,
    GRID_COLOR, GRID_ALPHA, TEXT_HEADER, TEXT_LABEL, TEXT_TICK, DARK_CMAP,
)
from loss_topography.surfaces import (
    Rosenbrock, RegularizedSaddle, TwinWell, MultiModal,
    StrictSaddle, Rastrigin, SURFACE_REGISTRY,
)
from loss_topography.optimizers import (
    SGD, MomentumSGD, NAG, Adam, AdamW, RMSProp, DampedNewton,
    get_default_optimizers,
)
from loss_topography.losses import ALL_LOSSES, MSE, MAE, Huber, LogCosh, Cauchy
from loss_topography.sharpness import (
    smooth_twin_well_1d, generate_shift_animation_data,
)
from loss_topography.benchmarks import run_full_benchmark, format_benchmark_table

ASSETS_DIR = os.path.join(ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

apply_mpl_theme()

DPI = 120
GIF_FPS = 20


# ═══════════════════════════════════════════════════════════════════
# 1. HERO OPTIMIZER RACE GIF
# ═══════════════════════════════════════════════════════════════════

def generate_hero_optimizer_race():
    print("  [1/6] Generating hero_optimizer_race.gif ...")

    surface = MultiModal()
    X, Y, Z = surface.meshgrid(n=300)

    # Clip Z for visualization
    Z_clip = np.clip(Z, np.percentile(Z, 2), np.percentile(Z, 95))

    # Run optimizers
    optimizers = {
        "SGD":      SGD(lr=0.05),
        "Momentum": MomentumSGD(lr=0.02, beta=0.9),
        "RMSProp":  RMSProp(lr=0.05),
        "Adam":     Adam(lr=0.08),
    }

    x0, y0 = 3.0, 3.0
    n_steps = 150
    logs = {}
    for name, opt in optimizers.items():
        logs[name] = opt.run(surface, x0, y0, n_steps=n_steps)

    # Set up figure: 2-panel (3D left, 2D contour right)
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)

    # Left: 3D surface
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax3d.set_facecolor(BG_CANVAS)
    ax3d.plot_surface(X, Y, Z_clip, cmap=DARK_CMAP, alpha=0.6,
                      rstride=4, cstride=4, linewidth=0, antialiased=True)
    ax3d.set_xlabel("θ₁", color=TEXT_LABEL, fontsize=10)
    ax3d.set_ylabel("θ₂", color=TEXT_LABEL, fontsize=10)
    ax3d.set_zlabel("Loss", color=TEXT_LABEL, fontsize=10)
    ax3d.set_title("3D Loss Surface", color=TEXT_HEADER, fontsize=12, pad=10)
    ax3d.tick_params(colors=TEXT_TICK, labelsize=7)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor(GRID_COLOR)
    ax3d.yaxis.pane.set_edgecolor(GRID_COLOR)
    ax3d.zaxis.pane.set_edgecolor(GRID_COLOR)
    ax3d.view_init(elev=35, azim=-55)

    # Right: 2D contour
    ax2d = fig.add_subplot(gs[1])
    ax2d.set_facecolor(BG_PANEL)
    levels = np.linspace(Z_clip.min(), Z_clip.max(), 40)
    ax2d.contourf(X, Y, Z_clip, levels=levels, cmap=DARK_CMAP, alpha=0.7)
    ax2d.contour(X, Y, Z_clip, levels=levels[::4], colors=[GRID_COLOR],
                 linewidths=0.3, alpha=0.5)
    ax2d.set_xlabel("θ₁", color=TEXT_LABEL)
    ax2d.set_ylabel("θ₂", color=TEXT_LABEL)
    ax2d.set_title("Optimizer Race (Contour View)", color=TEXT_HEADER, fontsize=12)
    ax2d.set_aspect("equal")

    # Animated elements
    trail_lines_3d = {}
    trail_lines_2d = {}
    markers_3d = {}
    markers_2d = {}
    label_texts = {}

    for name, color in [("SGD", OPTIMIZER_COLORS["SGD"]),
                         ("Momentum", OPTIMIZER_COLORS["Momentum"]),
                         ("RMSProp", OPTIMIZER_COLORS["RMSProp"]),
                         ("Adam", OPTIMIZER_COLORS["Adam"])]:
        trail_lines_3d[name], = ax3d.plot([], [], [], color=color, lw=1.5,
                                          alpha=0.8)
        markers_3d[name], = ax3d.plot([], [], [], 'o', color=color,
                                      markersize=5, zorder=10)
        trail_lines_2d[name], = ax2d.plot([], [], color=color, lw=1.8,
                                          alpha=0.9, label=name)
        markers_2d[name], = ax2d.plot([], [], 'o', color=color,
                                      markersize=7, zorder=10,
                                      markeredgecolor='white',
                                      markeredgewidth=0.5)

    ax2d.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # Leaderboard text
    leaderboard = ax2d.text(
        0.02, 0.02, "", transform=ax2d.transAxes,
        fontsize=7, family="monospace", color=TEXT_HEADER,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                  edgecolor=GRID_COLOR, alpha=0.9),
    )

    def animate(frame):
        idx = min(frame, n_steps)
        for name in logs:
            log = logs[name]
            xs = log.xs[:idx + 1]
            ys = log.ys[:idx + 1]
            zs = log.losses[:idx + 1]

            trail_lines_3d[name].set_data(xs, ys)
            trail_lines_3d[name].set_3d_properties(zs)
            markers_3d[name].set_data([xs[-1]], [ys[-1]])
            markers_3d[name].set_3d_properties([zs[-1]])

            trail_lines_2d[name].set_data(xs, ys)
            markers_2d[name].set_data([xs[-1]], [ys[-1]])

        # Leaderboard
        lines = [f"Step {idx:3d}"]
        for name in ["Adam", "RMSProp", "Momentum", "SGD"]:
            i = min(idx, len(logs[name].losses) - 1)
            lines.append(f"  {name:10s}  L={logs[name].losses[i]:.3f}")
        leaderboard.set_text("\n".join(lines))

        return []

    # Subsample frames by 2 for lightweight fast-playing GIF
    frame_list = list(range(0, n_steps + 1, 2))
    anim = FuncAnimation(fig, animate, frames=frame_list,
                         interval=50, blit=False)
    path = os.path.join(ASSETS_DIR, "hero_optimizer_race.gif")
    anim.save(path, writer=PillowWriter(fps=20), dpi=85)
    plt.close(fig)
    print(f"    ✓ Saved {path} ({os.path.getsize(path) / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════
# 2. SADDLE POINT ESCAPE GIF
# ═══════════════════════════════════════════════════════════════════

def generate_saddle_escape():
    print("  [2/6] Generating saddle_point_escape.gif ...")

    surface = RegularizedSaddle()
    X, Y, Z = surface.meshgrid(n=200)
    Z_clip = np.clip(Z, -3, 5)

    x0, y0 = 0.01, 0.001
    n_steps = 120

    optimizers = {
        "SGD":           SGD(lr=0.005),
        "Momentum":      MomentumSGD(lr=0.005, beta=0.95),
        "Adam":          Adam(lr=0.03),
        "Damped Newton": DampedNewton(lr=0.3, mu=0.5),
    }

    logs = {}
    for name, opt in optimizers.items():
        logs[name] = opt.run(surface, x0, y0, n_steps=n_steps)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    for ax in (ax_left, ax_right):
        ax.set_facecolor(BG_PANEL)
        levels = np.linspace(Z_clip.min(), Z_clip.max(), 35)
        ax.contourf(X, Y, Z_clip, levels=levels, cmap=DARK_CMAP, alpha=0.7)
        ax.contour(X, Y, Z_clip, levels=levels[::3], colors=[GRID_COLOR],
                   linewidths=0.3, alpha=0.5)
        ax.set_xlabel("θ₁", color=TEXT_LABEL)
        ax.set_ylabel("θ₂", color=TEXT_LABEL)
        ax.set_aspect("equal")

    ax_left.set_title("SGD (Stuck at Saddle)", color="#EF4444", fontsize=12)
    ax_right.set_title("Momentum / Adam / Newton (Escape!)",
                       color="#10B981", fontsize=12)

    # SGD on left
    sgd_trail, = ax_left.plot([], [], color=OPTIMIZER_COLORS["SGD"],
                               lw=2, alpha=0.9, label="SGD")
    sgd_marker, = ax_left.plot([], [], 'o', color=OPTIMIZER_COLORS["SGD"],
                                markersize=8, markeredgecolor="white",
                                markeredgewidth=1)

    # Others on right
    trails = {}
    markers = {}
    for name in ["Momentum", "Adam", "Damped Newton"]:
        color = OPTIMIZER_COLORS[name]
        trails[name], = ax_right.plot([], [], color=color, lw=2,
                                       alpha=0.9, label=name)
        markers[name], = ax_right.plot([], [], 'o', color=color,
                                        markersize=8, markeredgecolor="white",
                                        markeredgewidth=1)

    ax_left.legend(loc="upper left", fontsize=9)
    ax_right.legend(loc="upper left", fontsize=9)

    # Hessian eigenvalue annotation
    eig_text = ax_left.text(
        0.05, 0.95, "", transform=ax_left.transAxes,
        fontsize=9, color=TEXT_HEADER, verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor=BG_PANEL, edgecolor=GRID_COLOR,
                  alpha=0.9),
    )

    # Mark saddle point
    for ax in (ax_left, ax_right):
        ax.plot(0, 0, 'x', color="#FBBF24", markersize=12, markeredgewidth=2.5,
                zorder=20)
        ax.annotate("Saddle\n(0, 0)", (0, 0), textcoords="offset points",
                    xytext=(15, 15), color="#FBBF24", fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="#FBBF24", lw=1))

    def animate(frame):
        idx = min(frame, n_steps)
        log = logs["SGD"]
        sgd_trail.set_data(log.xs[:idx+1], log.ys[:idx+1])
        sgd_marker.set_data([log.xs[min(idx, len(log.xs)-1)]],
                            [log.ys[min(idx, len(log.ys)-1)]])

        for name in ["Momentum", "Adam", "Damped Newton"]:
            log = logs[name]
            trails[name].set_data(log.xs[:idx+1], log.ys[:idx+1])
            markers[name].set_data([log.xs[min(idx, len(log.xs)-1)]],
                                   [log.ys[min(idx, len(log.ys)-1)]])

        # Show Hessian eigenvalues at origin
        eig_text.set_text(
            f"Hessian at (0,0):\n"
            f"λ₁ = +2.0 (ridge)\n"
            f"λ₂ = −2.0 (escape)\n"
            f"Step: {idx}"
        )
        return []

    anim = FuncAnimation(fig, animate, frames=n_steps + 1,
                         interval=1000 // GIF_FPS, blit=False)
    path = os.path.join(ASSETS_DIR, "saddle_point_escape.gif")
    anim.save(path, writer=PillowWriter(fps=GIF_FPS), dpi=DPI)
    plt.close(fig)
    print(f"    ✓ Saved {path} ({os.path.getsize(path) / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════
# 3. SHARP VS FLAT MINIMA GIF
# ═══════════════════════════════════════════════════════════════════

def generate_sharp_vs_flat():
    print("  [3/6] Generating sharp_vs_flat_minima.gif ...")

    data = generate_shift_animation_data(n_frames=60, max_shift=0.8)
    theta = data["theta"]
    train_loss = data["train_loss"]
    n_frames = len(data["shifts"])

    fig, (ax_main, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.3},
    )

    ax_main.set_facecolor(BG_PANEL)
    ax_bar.set_facecolor(BG_PANEL)

    # Static training loss
    ax_main.plot(theta, train_loss, color="#06B6D4", lw=2, label="Train Loss",
                 zorder=5)

    # Mark minima
    ax_main.axvline(-2, color="#EF4444", ls="--", alpha=0.5, lw=1)
    ax_main.axvline(2, color="#10B981", ls="--", alpha=0.5, lw=1)
    ax_main.annotate("Sharp\nMin", (-2, 0.1), color="#EF4444", fontsize=10,
                     ha="center", fontweight="bold")
    ax_main.annotate("Flat\nMin", (2, 0.1), color="#10B981", fontsize=10,
                     ha="center", fontweight="bold")

    ax_main.set_xlabel("θ (parameter)", color=TEXT_LABEL)
    ax_main.set_ylabel("Loss", color=TEXT_LABEL)
    ax_main.set_title("Distribution Shift Wave", color=TEXT_HEADER, fontsize=13)
    ax_main.set_xlim(-5, 5)
    ax_main.set_ylim(-0.5, 5)

    # Animated shifted test loss curve
    shifted_line, = ax_main.plot([], [], color="#EC4899", lw=2, ls="--",
                                  alpha=0.8, label="Test Loss (shifted)")
    ax_main.legend(loc="upper right", fontsize=9)

    # Shift annotation
    shift_text = ax_main.text(
        0.02, 0.95, "", transform=ax_main.transAxes,
        fontsize=10, color=TEXT_HEADER, verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor=BG_PANEL, edgecolor=GRID_COLOR,
                  alpha=0.9),
    )

    # Bar chart for test losses
    bar_names = ["Sharp", "Flat"]
    bar_colors = ["#EF4444", "#10B981"]
    bars = ax_bar.bar(bar_names, [0, 0], color=bar_colors, edgecolor="white",
                      linewidth=0.5, width=0.6)
    ax_bar.set_ylabel("Test Loss", color=TEXT_LABEL)
    ax_bar.set_title("Test Loss Spike", color=TEXT_HEADER, fontsize=12)
    ax_bar.set_ylim(0, 5)

    def animate(frame):
        shifted_curve = data["shifted_curves"][frame]
        shifted_line.set_data(theta, shifted_curve)

        shift_val = data["shifts"][frame]
        shift_text.set_text(f"Δθ = {shift_val:.3f}")

        sharp_test = data["sharp_test_losses"][frame]
        flat_test = data["flat_test_losses"][frame]
        bars[0].set_height(sharp_test)
        bars[1].set_height(flat_test)

        return []

    anim = FuncAnimation(fig, animate, frames=n_frames,
                         interval=1000 // GIF_FPS, blit=False)
    path = os.path.join(ASSETS_DIR, "sharp_vs_flat_minima.gif")
    anim.save(path, writer=PillowWriter(fps=GIF_FPS), dpi=DPI)
    plt.close(fig)
    print(f"    ✓ Saved {path} ({os.path.getsize(path) / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════
# 4. LOSS FUNCTION ATLAS PNG
# ═══════════════════════════════════════════════════════════════════

def generate_loss_atlas():
    print("  [4/6] Generating loss_function_atlas.png ...")

    losses = ALL_LOSSES
    n_rows, n_cols = 2, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 8))

    r = np.linspace(-4, 4, 500)
    loss_colors = [
        "#06B6D4", "#F59E0B", "#10B981", "#A855F7", "#EC4899",
        "#3B82F6", "#EF4444", "#FBBF24", "#8B5CF6", "#F43F5E",
    ]

    for idx, (ax, loss) in enumerate(zip(axes.flat, losses)):
        ax.set_facecolor(BG_PANEL)
        color = loss_colors[idx % len(loss_colors)]

        # Value
        vals = loss.value(r)
        vals = np.clip(vals, -10, 10)
        ax.plot(r, vals, color=color, lw=2.0, label=r"$\mathcal{L}(r)$")

        # Gradient
        grads = loss.grad(r)
        grads = np.clip(grads, -10, 10)
        ax.plot(r, grads, color=color, lw=1.2, ls="--", alpha=0.6,
                label=r"$\nabla\mathcal{L}$")

        ax.set_title(loss.name, color=color, fontsize=11, fontweight="bold")
        ax.set_xlabel("r", color=TEXT_TICK, fontsize=8)
        ax.axhline(0, color=GRID_COLOR, lw=0.5, alpha=0.5)
        ax.axvline(0, color=GRID_COLOR, lw=0.5, alpha=0.5)
        ax.set_xlim(-4, 4)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)

    fig.suptitle("The Loss Function Atlas — Value & Gradient",
                 color=TEXT_HEADER, fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = os.path.join(ASSETS_DIR, "loss_function_atlas.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved {path} ({os.path.getsize(path) / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════
# 5. LOSS CONVERGENCE TIMING PNG
# ═══════════════════════════════════════════════════════════════════

def generate_convergence_timing():
    print("  [5/6] Generating loss_convergence_timing.png ...")

    # Run optimizers on Rosenbrock
    surface = Rosenbrock()
    x0, y0 = -1.5, 2.0
    n_steps = 300

    optimizers = {
        "SGD":      SGD(lr=0.001),
        "Momentum": MomentumSGD(lr=0.001, beta=0.9),
        "NAG":      NAG(lr=0.001, beta=0.9),
        "RMSProp":  RMSProp(lr=0.005),
        "Adam":     Adam(lr=0.01),
        "AdamW":    AdamW(lr=0.01),
    }

    logs = {}
    for name, opt in optimizers.items():
        logs[name] = opt.run(surface, x0, y0, n_steps=n_steps)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel (a): Loss vs Iteration (log scale)
    ax = axes[0, 0]
    ax.set_facecolor(BG_PANEL)
    for name, log in logs.items():
        color = OPTIMIZER_COLORS.get(name, "#FFFFFF")
        losses = np.clip(log.losses, 1e-8, None)
        ax.semilogy(losses, color=color, lw=1.8, label=name, alpha=0.9)
    ax.set_xlabel("Iteration", color=TEXT_LABEL)
    ax.set_ylabel("Loss (log scale)", color=TEXT_LABEL)
    ax.set_title("(a) Loss vs. Iteration", color=TEXT_HEADER, fontsize=12)
    ax.legend(fontsize=8)

    # Panel (b): Loss vs Wall-Clock Time
    ax = axes[0, 1]
    ax.set_facecolor(BG_PANEL)
    for name, log in logs.items():
        color = OPTIMIZER_COLORS.get(name, "#FFFFFF")
        cum_time = np.cumsum(log.wall_clocks) / 1000  # convert to ms
        losses = np.clip(log.losses, 1e-8, None)
        ax.semilogy(cum_time, losses, color=color, lw=1.8, label=name, alpha=0.9)
    ax.set_xlabel("Wall-Clock Time (ms)", color=TEXT_LABEL)
    ax.set_ylabel("Loss (log scale)", color=TEXT_LABEL)
    ax.set_title("(b) Loss vs. Wall-Clock Time", color=TEXT_HEADER, fontsize=12)
    ax.legend(fontsize=8)

    # Panel (c): Path Efficiency across surfaces
    ax = axes[1, 0]
    ax.set_facecolor(BG_PANEL)

    test_surfaces = ["Rosenbrock", "Rastrigin", "Regularized Saddle"]
    opt_names_short = ["SGD", "Momentum", "Adam", "RMSProp"]
    x_pos = np.arange(len(test_surfaces))
    width = 0.18

    for i, opt_name in enumerate(opt_names_short):
        efficiencies = []
        for sname in test_surfaces:
            s = SURFACE_REGISTRY[sname]
            opt_cls = optimizers.get(opt_name, SGD(lr=0.01))
            if isinstance(opt_cls, type):
                opt_inst = opt_cls()
            else:
                opt_inst = opt_cls
            opt_inst.reset()
            log = opt_inst.run(s, -1.5, 2.0, n_steps=200)
            efficiencies.append(log.path_efficiency)
        color = OPTIMIZER_COLORS.get(opt_name, "#FFFFFF")
        ax.bar(x_pos + i * width, efficiencies, width, color=color,
               label=opt_name, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels(test_surfaces, fontsize=8)
    ax.set_ylabel("Path Efficiency", color=TEXT_LABEL)
    ax.set_title("(c) Path Efficiency Ratio", color=TEXT_HEADER, fontsize=12)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Panel (d): Convergence steps bar chart
    ax = axes[1, 1]
    ax.set_facecolor(BG_PANEL)

    conv_steps = {}
    for name, log in logs.items():
        conv_steps[name] = log.converged_step if log.converged_step else n_steps

    names = list(conv_steps.keys())
    steps = list(conv_steps.values())
    colors = [OPTIMIZER_COLORS.get(n, "#FFFFFF") for n in names]
    ax.barh(names, steps, color=colors, edgecolor="white", linewidth=0.3, height=0.6)
    ax.set_xlabel("Steps to ε-Converge", color=TEXT_LABEL)
    ax.set_title("(d) Convergence Speed (Rosenbrock)", color=TEXT_HEADER, fontsize=12)

    fig.suptitle("Convergence & Timing Benchmark Dashboard",
                 color=TEXT_HEADER, fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()

    path = os.path.join(ASSETS_DIR, "loss_convergence_timing.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved {path} ({os.path.getsize(path) / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════
# 6. FILTER-NORMALIZED NN SURFACE PNG
# ═══════════════════════════════════════════════════════════════════

def generate_nn_surface():
    print("  [6/6] Generating filter_normalized_nn_surface.png ...")

    from loss_topography.nn_landscape import train_and_visualize_landscape

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.2, 1, 1.2, 1], wspace=0.35)

    for panel_idx, (use_skip, label) in enumerate(
            [(False, "Plain MLP (No Skip)"), (True, "Residual MLP (Skip Conn.)")]):

        result = train_and_visualize_landscape(
            use_skip=use_skip, grid_size=25, alpha_range=1.0,
            n_epochs=150, hidden=32, seed=42,
        )

        A = result["alpha_grid"]
        B = result["beta_grid"]
        Z = result["loss_surface"]
        Z_clip = np.clip(Z, np.percentile(Z, 1), np.percentile(Z, 99))

        # 3D surface
        ax3d = fig.add_subplot(gs[panel_idx * 2], projection="3d")
        ax3d.set_facecolor(BG_CANVAS)
        ax3d.plot_surface(A, B, Z_clip, cmap=DARK_CMAP, alpha=0.8,
                          rstride=1, cstride=1, linewidth=0, antialiased=True)
        ax3d.set_xlabel("α", color=TEXT_LABEL, fontsize=9)
        ax3d.set_ylabel("β", color=TEXT_LABEL, fontsize=9)
        ax3d.set_zlabel("Loss", color=TEXT_LABEL, fontsize=9)

        title_color = "#EF4444" if not use_skip else "#10B981"
        ax3d.set_title(label, color=title_color, fontsize=11, fontweight="bold")
        ax3d.tick_params(colors=TEXT_TICK, labelsize=6)
        ax3d.view_init(elev=30, azim=-50)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor(GRID_COLOR)
        ax3d.yaxis.pane.set_edgecolor(GRID_COLOR)
        ax3d.zaxis.pane.set_edgecolor(GRID_COLOR)

        # 2D contour
        ax2d = fig.add_subplot(gs[panel_idx * 2 + 1])
        ax2d.set_facecolor(BG_PANEL)
        levels = np.linspace(Z_clip.min(), Z_clip.max(), 30)
        cf = ax2d.contourf(A, B, Z_clip, levels=levels, cmap=DARK_CMAP)
        ax2d.contour(A, B, Z_clip, levels=levels[::3], colors=[GRID_COLOR],
                     linewidths=0.3, alpha=0.5)
        ax2d.set_xlabel("α", color=TEXT_LABEL, fontsize=9)
        ax2d.set_ylabel("β", color=TEXT_LABEL, fontsize=9)
        ax2d.set_title("Contour View", color=TEXT_HEADER, fontsize=10)
        ax2d.set_aspect("equal")

        # Mark center (trained weights)
        ax2d.plot(0, 0, '*', color="#FBBF24", markersize=12, zorder=10,
                  markeredgecolor="white", markeredgewidth=0.5)

    fig.suptitle("Filter-Normalized Neural Network Loss Landscapes (Li et al., 2018)",
                 color=TEXT_HEADER, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(ASSETS_DIR, "filter_normalized_nn_surface.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved {path} ({os.path.getsize(path) / 1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  loss-topography Asset Generator")
    print("=" * 60)
    print()

    generate_hero_optimizer_race()
    generate_saddle_escape()
    generate_sharp_vs_flat()
    generate_loss_atlas()
    generate_convergence_timing()
    generate_nn_surface()

    # Flowing river visualization & interactive 3D HTML
    try:
        from scripts.generate_river_flow import generate_optimization_river_flow
        generate_optimization_river_flow()
    except Exception as e:
        print(f"  ⚠ Skipping river flow: {e}")

    try:
        from scripts.export_interactive_html import export_interactive_html
        export_interactive_html()
    except Exception as e:
        print(f"  ⚠ Skipping interactive HTML: {e}")

    print()
    print("=" * 60)
    print("  All assets generated successfully!")
    print("=" * 60)

    # Verify all files exist and are non-empty
    expected = [
        "hero_optimizer_race.gif",
        "optimization_river_flow.gif",
        "saddle_point_escape.gif",
        "sharp_vs_flat_minima.gif",
        "loss_function_atlas.png",
        "loss_convergence_timing.png",
        "filter_normalized_nn_surface.png",
        "interactive_3d_landscape.html",
    ]
    all_ok = True
    for fname in expected:
        fpath = os.path.join(ASSETS_DIR, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            status = "✓" if size_kb > 10 else "⚠ (small)"
            print(f"  {status} {fname}: {size_kb:.0f} KB")
        else:
            print(f"  ✗ {fname}: MISSING!")
            all_ok = False

    if all_ok:
        print("\n  🎉 All visual assets and interactive artifacts verified!")
    else:
        print("\n  ❌ Some assets are missing!")
        sys.exit(1)
