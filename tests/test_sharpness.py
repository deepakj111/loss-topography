"""
Tests for sharpness, flat vs sharp minima, and distribution shift simulations.
"""

from __future__ import annotations
import pytest
import numpy as np
from loss_topography.sharpness import (
    smooth_twin_well_1d,
    compute_sharpness,
    generate_shift_animation_data,
    SharpnessResult,
)


def test_smooth_twin_well_shape():
    theta = np.linspace(-4, 4, 100)
    loss = smooth_twin_well_1d(theta)
    assert loss.shape == (100,)
    assert np.all(np.isfinite(loss))


def test_sharp_vs_flat_curvatures():
    def loss_func(th):
        return smooth_twin_well_1d(th, sharp_curv=50.0, flat_curv=1.0)

    sharp_res = compute_sharpness(loss_func, theta_star=-2.0, epsilon=0.4)
    flat_res = compute_sharpness(loss_func, theta_star=2.0, epsilon=0.4)

    assert isinstance(sharp_res, SharpnessResult)
    assert isinstance(flat_res, SharpnessResult)
    assert sharp_res.name == "Sharp"
    assert flat_res.name == "Flat"

    # Sharp well should have higher curvature (Hessian) and higher epsilon sharpness
    assert sharp_res.hessian_trace > flat_res.hessian_trace
    assert sharp_res.epsilon_sharpness > flat_res.epsilon_sharpness

    # Under distribution shift (+0.3), sharp well penalty should be greater than flat well penalty
    assert sharp_res.shifted_test_loss > flat_res.shifted_test_loss


def test_generate_shift_animation_data():
    anim_data = generate_shift_animation_data(n_frames=10, max_shift=0.4)
    assert "theta" in anim_data
    assert "train_loss" in anim_data
    assert "shifts" in anim_data
    assert "shifted_curves" in anim_data
    assert len(anim_data["shifts"]) == 10
    assert len(anim_data["shifted_curves"]) == 10
    assert len(anim_data["sharp_test_losses"]) == 10
    assert len(anim_data["flat_test_losses"]) == 10
    assert anim_data["sharp_test_losses"][-1] > anim_data["flat_test_losses"][-1]
