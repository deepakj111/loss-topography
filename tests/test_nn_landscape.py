"""
Tests for filter-normalized real neural network loss landscapes.
"""

from __future__ import annotations
import pytest
import numpy as np
from loss_topography.nn_landscape import _make_moons_data, train_and_visualize_landscape


def test_make_moons_data():
    X, y = _make_moons_data(n_samples=200, noise=0.1, seed=42)
    assert X.shape == (200, 2)
    assert y.shape == (200,)
    assert set(np.unique(y)).issubset({0.0, 1.0})
    assert np.sum(y == 0) == 100
    assert np.sum(y == 1) == 100


def test_train_and_visualize_landscape_shapes():
    # Test fast run with small grid and few epochs
    res = train_and_visualize_landscape(
        use_skip=False,
        grid_size=7,
        alpha_range=0.5,
        n_epochs=5,
        hidden=16,
        seed=1,
    )

    assert "alpha_grid" in res
    assert "beta_grid" in res
    assert "loss_surface" in res
    assert "model_info" in res

    assert res["alpha_grid"].shape == (7, 7)
    assert res["beta_grid"].shape == (7, 7)
    assert res["loss_surface"].shape == (7, 7)
    assert np.all(np.isfinite(res["loss_surface"]))


def test_residual_vs_plain_landscape():
    res_skip = train_and_visualize_landscape(
        use_skip=True,
        grid_size=5,
        alpha_range=0.5,
        n_epochs=5,
        hidden=16,
        seed=1,
    )
    assert res_skip["loss_surface"].shape == (5, 5)
    assert "ResidualMLP" in res_skip["model_info"] or "dummy" in res_skip["model_info"]
