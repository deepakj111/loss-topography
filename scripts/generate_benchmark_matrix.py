#!/usr/bin/env python3
"""
Comprehensive Multi-Surface Optimizer Benchmark Suite & Matrix Visualizer.

Evaluates 8 optimizers (from basic gradient descent to 2nd-order Newton)
across 5 benchmark loss landscapes, generating:
  1. assets/optimizer_benchmark_matrix.png  (4-panel visual dashboard)
  2. Aggregated Markdown comparison table for the README
"""

from __future__ import annotations
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loss_topography.theme import (
    apply_mpl_theme, OPTIMIZER_COLORS, BG_CANVAS, BG_PANEL,
    GRID_COLOR, TEXT_HEADER, TEXT_LABEL, TEXT_TICK,
)
from loss_topography.surfaces import (
    Rosenbrock, Rastrigin, RegularizedSaddle, Beale, MultiModal,
)
from loss_topography.optimizers import (
    SGD, MomentumSGD, NAG, AdaGrad, RMSProp, Adam, AdamW, DampedNewton,
)
from loss_topography.benchmarks import run_benchmark, BenchmarkResult

ASSETS_DIR = os.path.join(ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
apply_mpl_theme()


def run_comprehensive_suite():
    print("=" * 70)
    print("  loss-topography — Multi-Surface Optimizer Benchmark Suite")
    print("=" * 70)

    # 5 diverse landscapes testing different geometric difficulties
    surfaces = [
        ("Beale", Beale(), 2.0, 2.0, 0.0),
        ("Rosenbrock", Rosenbrock(), -0.8, 1.2, 0.0),
        ("Rastrigin", Rastrigin(), 2.5, 2.5, 0.0),
        ("Regularized Saddle", RegularizedSaddle(), 0.05, 0.1, -2.5),
        ("Multi-Modal", MultiModal(), 2.5, 2.5, -6.55),
    ]

    # 8 optimizers from basic first-order to second-order
    def make_optimizers():
        return {
            "SGD":           SGD(lr=0.01),
            "Momentum":      MomentumSGD(lr=0.01, beta=0.85),
            "NAG":           NAG(lr=0.01, beta=0.85),
            "AdaGrad":       AdaGrad(lr=0.35),
            "RMSProp":       RMSProp(lr=0.02),
            "Adam":          Adam(lr=0.04),
            "AdamW":         AdamW(lr=0.04),
            "Damped Newton": DampedNewton(lr=0.5, mu=0.8),
        }

    n_steps = 250
    results_by_surf = {}
    all_results = []

    for sname, surf, x0, y0, f_star in surfaces:
        print(f"\n  Running benchmarks on {sname} (start: {x0}, {y0}, f*={f_star})...")
        opts = make_optimizers()
        surf_res = []
        for oname, opt in opts.items():
            res = run_benchmark(surf, opt, x0, y0, n_steps=n_steps, f_star=f_star)
            surf_res.append(res)
            all_results.append(res)
            conv_str = f"step {res.steps_to_converge}" if res.steps_to_converge else "did not reach ε"
            print(f"    • {oname:15s} | Final Loss: {res.final_loss:9.4f} | Eff: {res.path_efficiency:.3f} | {conv_str}")
        results_by_surf[sname] = surf_res

    # ── Create 4-panel visual dashboard ──────────────────────────
    print("\n  Generating assets/optimizer_benchmark_matrix.png ...")
    fig = plt.figure(figsize=(15, 9.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.22)

    opt_names = list(make_optimizers().keys())
    surf_names = [s[0] for s in surfaces]

    # Panel 1: Trajectory Path Efficiency Across Surfaces (Grouped Bar)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG_PANEL)
    x_indices = np.arange(len(surf_names))
    width = 0.095

    for i, oname in enumerate(opt_names):
        effs = []
        for sname in surf_names:
            matching = [r for r in results_by_surf[sname] if r.optimizer_name == oname][0]
            effs.append(matching.path_efficiency)
        offset = (i - len(opt_names) / 2 + 0.5) * width
        color = OPTIMIZER_COLORS.get(oname, "#FFFFFF")
        ax1.bar(x_indices + offset, effs, width, label=oname, color=color, alpha=0.9, edgecolor="#000000", lw=0.4)

    ax1.set_title("(a) Trajectory Path Efficiency (Higher = Straighter Geodesic)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(surf_names, fontsize=8.5, color=TEXT_LABEL)
    ax1.set_ylabel("Path Efficiency Ratio ∈ (0, 1]", color=TEXT_LABEL, fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=7.5, ncol=2, framealpha=0.85)

    # Panel 2: Steps to Convergence (Bar Chart) on Rosenbrock & Regularized Saddle
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG_PANEL)
    y_pos = np.arange(len(opt_names))

    # Compare on Rosenbrock (ill-conditioned valley) and Regularized Saddle (escape)
    rosen_steps = []
    saddle_steps = []
    for oname in opt_names:
        r_res = [r for r in results_by_surf["Rosenbrock"] if r.optimizer_name == oname][0]
        s_res = [r for r in results_by_surf["Regularized Saddle"] if r.optimizer_name == oname][0]
        # If not converged, represent as max steps
        rosen_steps.append(r_res.steps_to_converge if r_res.steps_to_converge else n_steps)
        saddle_steps.append(s_res.steps_to_converge if s_res.steps_to_converge else n_steps)

    b_width = 0.35
    ax2.barh(y_pos - b_width/2, rosen_steps, b_width, label="Rosenbrock Valley", color="#F59E0B", alpha=0.85)
    ax2.barh(y_pos + b_width/2, saddle_steps, b_width, label="Regularized Saddle", color="#38BDF8", alpha=0.85)
    ax2.axvline(n_steps, color="#EF4444", linestyle=":", lw=1.2, label="Budget Limit (250)")

    ax2.set_title("(b) Steps to Reach ε-Convergence (Lower = Faster Progress)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(opt_names, fontsize=8.5, color=TEXT_LABEL)
    ax2.set_xlabel("Steps (capped at 250)", color=TEXT_LABEL, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # Panel 3: Computational Overhead (µs per gradient step)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG_PANEL)
    mean_latencies = []
    for oname in opt_names:
        lats = [r.mean_step_us for r in all_results if r.optimizer_name == oname]
        mean_latencies.append(np.mean(lats))

    bar_colors = [OPTIMIZER_COLORS.get(name, "#38BDF8") for name in opt_names]
    bars = ax3.barh(y_pos, mean_latencies, height=0.55, color=bar_colors, alpha=0.9, edgecolor="#000000", lw=0.4)
    for bar, lat in zip(bars, mean_latencies):
        ax3.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, f"{lat:.1f} µs",
                 va="center", ha="left", color=TEXT_TICK, fontsize=7.5)

    ax3.set_title("(c) Computational Overhead: Wall-Clock Latency per Step", fontsize=11, fontweight="bold", pad=10)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(opt_names, fontsize=8.5, color=TEXT_LABEL)
    ax3.set_xlabel("Latency (µs / step)", color=TEXT_LABEL, fontsize=9)
    ax3.set_xlim(0, max(mean_latencies) * 1.25)
    ax3.grid(True, linestyle="--", alpha=0.3)

    # Panel 4: Ranking Heatmap across all 5 surfaces
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(BG_PANEL)

    # Compute ranks on each surface based on final loss
    rank_matrix = np.zeros((len(opt_names), len(surf_names)))
    for j, sname in enumerate(surf_names):
        surf_losses = [(r.optimizer_name, r.final_loss) for r in results_by_surf[sname]]
        surf_losses.sort(key=lambda item: item[1])
        for rank, (oname, _) in enumerate(surf_losses):
            i = opt_names.index(oname)
            rank_matrix[i, j] = rank + 1  # 1st to 8th

    im = ax4.imshow(rank_matrix, cmap="viridis_r", aspect="auto", vmin=1, vmax=8)
    ax4.set_title("(d) Relative Performance Matrix (1 = Best/Gold, 8 = Lowest)", fontsize=11, fontweight="bold", pad=10)
    ax4.set_xticks(np.arange(len(surf_names)))
    ax4.set_xticklabels(surf_names, fontsize=8.5, color=TEXT_LABEL)
    ax4.set_yticks(np.arange(len(opt_names)))
    ax4.set_yticklabels(opt_names, fontsize=8.5, color=TEXT_LABEL)

    # Annotate rank numbers on cells
    for i in range(len(opt_names)):
        for j in range(len(surf_names)):
            r_val = int(rank_matrix[i, j])
            txt_color = "#FFFFFF" if r_val > 4 else "#000000"
            star = " ★" if r_val == 1 else ""
            ax4.text(j, i, f"#{r_val}{star}", ha="center", va="center",
                     color=txt_color, fontsize=8.5, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax4, shrink=0.7, pad=0.04)
    cbar.set_label("Rank (1=Best)", color=TEXT_LABEL, fontsize=8)
    cbar.ax.tick_params(colors=TEXT_TICK, labelsize=7)

    fig.suptitle("loss-topography — Multi-Surface Optimizer Benchmark Dashboard",
                 fontsize=14, fontweight="bold", color=TEXT_HEADER, y=0.98)

    save_path = os.path.join(ASSETS_DIR, "optimizer_benchmark_matrix.png")
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved {save_path} ({os.path.getsize(save_path) / 1024:.0f} KB)")

    # ── Print Aggregated Markdown Table ───────────────────────────
    print("\n" + "=" * 70)
    print("  Aggregated Benchmark Results Table (Ready for README)")
    print("=" * 70 + "\n")

    md_table = generate_aggregated_markdown_table(results_by_surf, surf_names, opt_names)
    print(md_table)
    return results_by_surf, md_table


def generate_aggregated_markdown_table(results_by_surf, surf_names, opt_names):
    lines = []
    lines.append("| Optimizer | Category | Beale | Rosenbrock | Rastrigin | Reg. Saddle | Multi-Modal | Mean Eff. | Mean Latency |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    categories = {
        "SGD": "1st-Order Basic",
        "Momentum": "Polyak Momentum",
        "NAG": "Nesterov Lookahead",
        "AdaGrad": "Historical Accum.",
        "RMSProp": "Moving Average",
        "Adam": "Adaptive Moments",
        "AdamW": "Decoupled Weight Decay",
        "Damped Newton": "2nd-Order Curvature",
    }

    for oname in opt_names:
        row_entries = [f"**{oname}**", categories[oname]]
        effs = []
        lats = []
        for sname in ["Beale", "Rosenbrock", "Rastrigin", "Regularized Saddle", "Multi-Modal"]:
            res = [r for r in results_by_surf[sname] if r.optimizer_name == oname][0]
            effs.append(res.path_efficiency)
            lats.append(res.mean_step_us)
            # Format: steps or final loss
            if res.steps_to_converge is not None:
                cell = f"**{res.steps_to_converge} st**"
            else:
                cell = f"{res.final_loss:.2f}"
            row_entries.append(cell)
        row_entries.append(f"{np.mean(effs):.3f}")
        row_entries.append(f"{np.mean(lats):.1f} µs")
        lines.append("| " + " | ".join(row_entries) + " |")

    return "\n".join(lines)


if __name__ == "__main__":
    run_comprehensive_suite()
