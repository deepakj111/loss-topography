"""
Quantitative Convergence Speed, Wall-Clock Timing, Oscillation Index,
and Condition-Number Analyzer.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .optimizers import (
    Optimizer, SGD, MomentumSGD, NAG, AdaGrad, RMSProp, Adam, AdamW,
    DampedNewton, TrajectoryLog, get_default_optimizers,
)
from .surfaces import Surface, SURFACE_REGISTRY


@dataclass
class BenchmarkResult:
    """Results from a single optimizer-surface benchmark."""
    optimizer_name: str
    surface_name: str
    steps_to_converge: Optional[int]
    final_loss: float
    mean_step_us: float        # mean wall-clock µs per step
    path_efficiency: float
    mean_condition_number: float
    oscillation_index: float   # std of loss differences / mean loss
    total_arc_length: float


def compute_oscillation_index(losses: np.ndarray) -> float:
    """Measure oscillation: std of consecutive loss differences / mean |loss|."""
    if len(losses) < 3:
        return 0.0
    diffs = np.diff(losses)
    mean_abs = np.mean(np.abs(losses[1:])) + 1e-12
    return float(np.std(diffs) / mean_abs)


def compute_arc_length(xs: np.ndarray, ys: np.ndarray) -> float:
    """Total path length traveled."""
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))


def run_benchmark(
    surface: Surface,
    optimizer: Optimizer,
    x0: float, y0: float,
    n_steps: int = 300,
    f_star: Optional[float] = None,
) -> BenchmarkResult:
    """Run a single optimizer on a surface and collect metrics."""
    log = optimizer.run(surface, x0, y0, n_steps=n_steps, f_star=f_star)

    losses = log.losses
    arc_len = compute_arc_length(log.xs, log.ys)
    osc = compute_oscillation_index(losses)
    mean_cond = float(np.mean([p.condition_number for p in log.points
                                if np.isfinite(p.condition_number)]))

    return BenchmarkResult(
        optimizer_name=optimizer.name,
        surface_name=surface.name,
        steps_to_converge=log.converged_step,
        final_loss=float(losses[-1]),
        mean_step_us=float(np.mean(log.wall_clocks)),
        path_efficiency=log.path_efficiency,
        mean_condition_number=mean_cond,
        oscillation_index=osc,
        total_arc_length=arc_len,
    )


def run_full_benchmark(
    surfaces: Optional[List[str]] = None,
    n_steps: int = 300,
    x0: float = -2.0,
    y0: float = 2.0,
) -> List[BenchmarkResult]:
    """Run all optimizers on selected surfaces."""
    if surfaces is None:
        surfaces = ["Rosenbrock", "Rastrigin", "Regularized Saddle",
                     "Multi-Modal Arena", "Beale"]

    results = []
    optimizers = get_default_optimizers()

    for surf_name in surfaces:
        surface = SURFACE_REGISTRY[surf_name]
        for opt_name, opt in optimizers.items():
            r = run_benchmark(surface, opt, x0, y0, n_steps=n_steps)
            results.append(r)

    return results


def format_benchmark_table(results: List[BenchmarkResult]) -> str:
    """Format results as a markdown table."""
    lines = [
        "| Surface | Optimizer | Steps→ε | Final Loss | µs/step | Path Eff. | Oscillation |",
        "|---------|-----------|---------|------------|---------|-----------|-------------|",
    ]
    for r in results:
        conv = str(r.steps_to_converge) if r.steps_to_converge is not None else "—"
        lines.append(
            f"| {r.surface_name[:15]:15s} | {r.optimizer_name:13s} | "
            f"{conv:>7s} | {r.final_loss:10.4f} | {r.mean_step_us:7.1f} | "
            f"{r.path_efficiency:9.4f} | {r.oscillation_index:11.4f} |"
        )
    return "\n".join(lines)
