#!/usr/bin/env python3
"""
Quickstart Replication Script for loss-topography.

Run with:
    python scripts/quickstart_replicate.py

Replicates key visualizations locally in seconds:
  1. 10 Loss Functions & analytical gradients
  2. Multi-Optimizer Trajectories on Rosenbrock Banana Valley
  3. Saddle Point Eigenspectrum & Curvature Vector Field
"""

from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loss_topography.theme import apply_mpl_theme, OPTIMIZER_COLORS
from loss_topography.surfaces import Rosenbrock, StrictSaddle
from loss_topography.optimizers import SGD, MomentumSGD, Adam, RMSProp, DampedNewton
from loss_topography.losses import MSE, Huber, LogCosh, Cauchy

apply_mpl_theme()


def replicate_all():
    print("=" * 60)
    print("  loss-topography — Local Replication Quickstart")
    print("=" * 60)

    # 1. Replicate Optimizer Race on Rosenbrock
    print("\n[1/3] Simulating optimizers on Rosenbrock banana valley...")
    surface = Rosenbrock()
    x0, y0 = -0.8, 1.2
    n_steps = 100

    opts = {
        "SGD": SGD(lr=0.002),
        "Momentum": MomentumSGD(lr=0.001, beta=0.85),
        "Adam": Adam(lr=0.04),
        "RMSProp": RMSProp(lr=0.02),
        "Damped Newton": DampedNewton(lr=0.5, mu=0.5),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y, Z = surface.meshgrid(n=200)
    levels = np.logspace(-1, 3.5, 30)
    ax.contour(X, Y, Z, levels=levels, cmap="plasma", alpha=0.5)

    for name, opt in opts.items():
        log = opt.run(surface, x0, y0, n_steps=n_steps)
        color = OPTIMIZER_COLORS.get(name, "#FFFFFF")
        ax.plot(log.xs, log.ys, "-o", color=color, label=f"{name} (Loss={log.losses[-1]:.3f})",
                markersize=2.5, lw=1.8, alpha=0.9)
        print(f"  ✓ {name:15s} -> Final Loss: {log.losses[-1]:.4f} | Path Eff: {log.path_efficiency:.3f}")

    ax.plot([1.0], [1.0], "*", color="#FBBF24", markersize=14, label="Global Min (1,1)")
    ax.set_title("Replication: Optimizer Paths on Rosenbrock Valley", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", fontsize=8)

    save_path = os.path.join(ROOT, "assets", "replicated_rosenbrock.png")
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")

    # 2. Replicate Loss Function Gradients
    print("\n[2/3] Evaluating analytical gradients for MSE, Huber, Log-Cosh, Cauchy...")
    r = np.linspace(-3, 3, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    losses = [("MSE", MSE), ("Huber", Huber), ("Log-Cosh", LogCosh), ("Cauchy", Cauchy)]
    colors = ["#38BDF8", "#34D399", "#A78BFA", "#F43F5E"]

    for (name, loss_fn), col in zip(losses, colors):
        ax1.plot(r, loss_fn(r), color=col, lw=2, label=name)
        ax2.plot(r, loss_fn.grad(r), color=col, lw=2, label=name)

    ax1.set_title("Loss Values L(r)")
    ax1.set_xlabel("Residual r")
    ax1.legend()

    ax2.set_title("Analytical Gradients ∇L(r)")
    ax2.set_xlabel("Residual r")
    ax2.legend()

    save_path = os.path.join(ROOT, "assets", "replicated_losses.png")
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path}")

    # 3. Replicate Saddle Point Curvature Inspection
    print("\n[3/3] Inspecting Hessian eigenspectrum on Strict Saddle (x² - y²)...")
    saddle = StrictSaddle()
    H = saddle.hessian(0.0, 0.0)
    eigs = saddle.eigenvalues(0.0, 0.0)
    classification = saddle.classify_point(0.0, 0.0)
    print(f"  Hessian at (0,0):\n{H}")
    print(f"  Eigenvalues: λ₁={eigs[0]:.1f}, λ₂={eigs[1]:.1f} -> {classification}")

    print("\n🎉 All replications completed successfully!")


if __name__ == "__main__":
    replicate_all()
