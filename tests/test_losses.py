"""Tests for all loss functions: finite-difference gradient & Hessian checks."""

import numpy as np
import pytest
from loss_topography.losses import (
    ALL_LOSSES, MSE, MAE, Huber, LogCosh, Quantile, BCE, Focal, Hinge,
    KL, Cauchy, numerical_grad, numerical_hess,
)


# Test points (avoid exact zero for losses with discontinuities)
TEST_POINTS = np.array([-2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0])

# Losses where analytical gradient is smooth enough for FD checks
SMOOTH_LOSSES = [MSE, LogCosh, Cauchy, BCE]


class TestLossValues:
    """Verify loss functions return reasonable values."""

    @pytest.mark.parametrize("loss", ALL_LOSSES)
    def test_non_negative_at_common_points(self, loss):
        """Most losses should be non-negative for typical inputs."""
        r = np.array([0.0, 0.5, 1.0, -0.5, -1.0])
        vals = loss.value(r)
        # Not all losses are non-negative everywhere (KL can be 0 at p=q)
        assert np.all(np.isfinite(vals)), f"{loss.name} produced non-finite values"

    @pytest.mark.parametrize("loss", ALL_LOSSES)
    def test_output_shape(self, loss):
        """Output shape should match input shape."""
        r = np.linspace(-3, 3, 50)
        assert loss.value(r).shape == r.shape
        assert loss.grad(r).shape == r.shape
        assert loss.hessian(r).shape == r.shape


class TestGradients:
    """Finite-difference gradient checks."""

    @pytest.mark.parametrize("loss", SMOOTH_LOSSES)
    def test_gradient_matches_finite_diff(self, loss):
        """Analytical gradient should match central FD to < 1e-5."""
        r = TEST_POINTS.copy()
        analytical = loss.grad(r)
        numerical = numerical_grad(loss, r, eps=1e-6)
        np.testing.assert_allclose(
            analytical, numerical, atol=1e-4, rtol=1e-4,
            err_msg=f"Gradient mismatch for {loss.name}",
        )

    def test_mse_gradient_exact(self):
        """MSE gradient should be exactly r."""
        r = np.array([-1.0, 0.0, 0.5, 2.0])
        np.testing.assert_array_equal(MSE.grad(r), r)

    def test_mae_gradient_sign(self):
        """MAE gradient should be sign(r)."""
        r = np.array([-2.0, -0.5, 0.5, 3.0])
        np.testing.assert_array_equal(MAE.grad(r), np.sign(r))


class TestHessians:
    """Finite-difference Hessian checks."""

    @pytest.mark.parametrize("loss", SMOOTH_LOSSES)
    def test_hessian_matches_finite_diff(self, loss):
        """Analytical Hessian should match central FD to < 1e-4."""
        r = TEST_POINTS.copy()
        analytical = loss.hessian(r)
        numerical = numerical_hess(loss, r, eps=1e-4)
        np.testing.assert_allclose(
            analytical, numerical, atol=1e-3, rtol=1e-3,
            err_msg=f"Hessian mismatch for {loss.name}",
        )

    def test_mse_hessian_constant(self):
        """MSE Hessian should be 1 everywhere."""
        r = np.linspace(-5, 5, 20)
        np.testing.assert_array_equal(MSE.hessian(r), np.ones_like(r))

    def test_logcosh_hessian_positive(self):
        """Log-Cosh Hessian (sech²) should always be positive."""
        r = np.linspace(-10, 10, 100)
        assert np.all(LogCosh.hessian(r) > 0)
