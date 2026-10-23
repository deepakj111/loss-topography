#!/usr/bin/env python3
"""
CLI benchmark tool: prints rich terminal table of optimizer & loss convergence metrics.
"""

from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loss_topography.benchmarks import run_full_benchmark, format_benchmark_table


def main():
    print("=" * 70)
    print("  loss-topography — Optimizer Convergence Benchmark")
    print("=" * 70)
    print()

    results = run_full_benchmark(
        surfaces=["Rosenbrock", "Rastrigin", "Regularized Saddle"],
        n_steps=300,
        x0=-0.5,
        y0=1.0,
    )

    table = format_benchmark_table(results)
    print(table)
    print()

    # Summary statistics
    for surf in ["Rosenbrock", "Rastrigin", "Regularized Saddle"]:
        surf_results = [r for r in results if r.surface_name == surf]
        best = min(surf_results, key=lambda r: r.final_loss)
        print(f"  Best on {surf}: {best.optimizer_name} "
              f"(loss={best.final_loss:.6f}, eff={best.path_efficiency:.4f})")

    return table


if __name__ == "__main__":
    main()
