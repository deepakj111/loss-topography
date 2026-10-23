"""
Sharp vs. Flat Minima — Generalization Gap & Distribution Shift Simulator.

Implements the classic visualization from Hochreiter & Schmidhuber (1997)
and Keskar et al. (2017) showing why flat minima generalize better.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SharpnessResult:
    """Metrics for a minimum's sharpness."""
    name: str
    center: float           # θ*
    train_loss: float       # L(θ*)
    hessian_trace: float    # Tr(H)
    hessian_max_eig: float  # λ_max(H)
    epsilon_sharpness: float  # max_{||δ||≤ε} L(θ*+δ) - L(θ*)
    shifted_test_loss: float  # L(θ* + Δθ) under distribution shift


def twin_well_1d(theta: np.ndarray, sharpness_ratio: float = 40.0) -> np.ndarray:
    """
    1D twin-well potential with one sharp and one flat minimum.

    Sharp minimum at θ = -2  (curvature = sharpness_ratio)
    Flat minimum  at θ = +2  (curvature = 1)
    Both minima at loss = 0.
    """
    # Sharp well: sharpness_ratio * (θ + 2)² centered at -2
    sharp = sharpness_ratio * (theta + 2) ** 2
    # Flat well: (θ - 2)² centered at +2
    flat = (theta - 2) ** 2
    # Combine: take the lower envelope via soft-min
    # But for visualization, we use a smooth composite:
    barrier = 3.0 * np.exp(-0.3 * theta ** 2)
    loss = np.minimum(sharp, flat)  # piecewise
    # Smoother version:
    loss = (-sharpness_ratio * np.exp(-0.5 * sharpness_ratio * (theta + 2) ** 2)
            - 1.0 * np.exp(-0.5 * 1.0 * (theta - 2) ** 2)
            + barrier + 3.0)
    return loss


def smooth_twin_well_1d(theta: np.ndarray,
                        sharp_curv: float = 40.0,
                        flat_curv: float = 1.0) -> np.ndarray:
    """Smooth twin-well with exact curvature control."""
    # Use Gaussian wells with curvature = width⁻²
    sharp_width = 1.0 / np.sqrt(sharp_curv)
    flat_width = 1.0 / np.sqrt(flat_curv)

    sharp_well = -3.0 * np.exp(-0.5 * ((theta + 2) / sharp_width) ** 2)
    flat_well = -3.0 * np.exp(-0.5 * ((theta - 2) / flat_width) ** 2)
    barrier = 1.5 * np.exp(-0.3 * theta ** 2)

    return sharp_well + flat_well + barrier + 3.5


def compute_sharpness(loss_fn, theta_star: float, epsilon: float = 0.5,
                      n_samples: int = 1000) -> SharpnessResult:
    """Compute sharpness metrics for a 1D minimum."""
    # ε-sharpness: max perturbation within ball
    deltas = np.linspace(-epsilon, epsilon, n_samples)
    losses = loss_fn(theta_star + deltas)
    loss_at_star = float(loss_fn(np.array([theta_star]))[0])
    eps_sharpness = float(np.max(losses) - loss_at_star)

    # Hessian via finite differences
    h = 1e-5
    diff = (
        loss_fn(np.array([theta_star + h])) -
        2 * loss_fn(np.array([theta_star])) +
        loss_fn(np.array([theta_star - h]))
    ) / h ** 2
    hessian_val = float(np.asarray(diff).ravel()[0])

    # Distribution shift
    shift = 0.3
    shifted_loss = float(loss_fn(np.array([theta_star + shift]))[0])

    name = "Sharp" if hessian_val > 10 else "Flat"
    return SharpnessResult(
        name=name,
        center=theta_star,
        train_loss=loss_at_star,
        hessian_trace=hessian_val,
        hessian_max_eig=hessian_val,
        epsilon_sharpness=eps_sharpness,
        shifted_test_loss=shifted_loss,
    )


def generate_shift_animation_data(
    n_frames: int = 60,
    max_shift: float = 0.5,
    sharp_curv: float = 40.0,
    flat_curv: float = 1.0,
) -> dict:
    """Generate data for the sharp vs flat animation."""
    theta = np.linspace(-5, 5, 500)

    def loss_fn(t):
        return smooth_twin_well_1d(t, sharp_curv, flat_curv)

    train_loss = loss_fn(theta)

    # Find minima
    sharp_center = -2.0
    flat_center = 2.0

    shifts = np.linspace(0, max_shift, n_frames)
    sharp_test_losses = []
    flat_test_losses = []
    shifted_curves = []

    for shift in shifts:
        test_curve = loss_fn(theta - shift)  # shift distribution
        shifted_curves.append(test_curve)

        sharp_test = float(loss_fn(np.array([sharp_center + shift]))[0])
        flat_test = float(loss_fn(np.array([flat_center + shift]))[0])
        sharp_test_losses.append(sharp_test)
        flat_test_losses.append(flat_test)

    return {
        "theta": theta,
        "train_loss": train_loss,
        "shifts": shifts,
        "shifted_curves": shifted_curves,
        "sharp_center": sharp_center,
        "flat_center": flat_center,
        "sharp_test_losses": np.array(sharp_test_losses),
        "flat_test_losses": np.array(flat_test_losses),
        "sharp_train_loss": float(loss_fn(np.array([sharp_center]))[0]),
        "flat_train_loss": float(loss_fn(np.array([flat_center]))[0]),
    }
