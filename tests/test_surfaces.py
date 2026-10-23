"""Tests for surfaces: critical point & eigenvalue verification."""

import numpy as np
import pytest
from loss_topography.surfaces import (
    StrictSaddle, MonkeySaddle, Rosenbrock, Rastrigin,
    RegularizedSaddle, TwinWell, MultiModal, ALL_SURFACES,
)


class TestCriticalPoints:
    """Verify critical points have zero gradient and correct Hessian eigenvalue signs."""

    def test_strict_saddle_origin(self):
        s = StrictSaddle()
        gx, gy = s.grad(np.array(0.0), np.array(0.0))
        assert abs(float(gx)) < 1e-10
        assert abs(float(gy)) < 1e-10

        eigvals, _ = s.hessian_eig(0.0, 0.0)
        # Saddle: one positive, one negative eigenvalue
        assert eigvals[0] < 0 and eigvals[1] > 0, \
            f"Expected mixed signs, got {eigvals}"

    def test_monkey_saddle_degenerate(self):
        s = MonkeySaddle()
        H = s.hessian(0.0, 0.0)
        # At origin, H should be zero matrix
        np.testing.assert_allclose(H, np.zeros((2, 2)), atol=1e-10)

    def test_rosenbrock_global_min(self):
        s = Rosenbrock()
        # Global min at (1, 1) with f = 0
        val = s.f_scalar(1.0, 1.0)
        assert abs(val) < 1e-10
        gx, gy = s.grad(np.array(1.0), np.array(1.0))
        assert abs(float(gx)) < 1e-6
        assert abs(float(gy)) < 1e-6

        # Hessian should be positive definite at the minimum
        eigvals, _ = s.hessian_eig(1.0, 1.0)
        assert np.all(eigvals > 0), f"Expected PD Hessian, got eigvals {eigvals}"

    def test_rastrigin_global_min(self):
        s = Rastrigin()
        val = s.f_scalar(0.0, 0.0)
        assert abs(val) < 1e-10
        gx, gy = s.grad(np.array(0.0), np.array(0.0))
        assert abs(float(gx)) < 1e-10
        assert abs(float(gy)) < 1e-10

    def test_regularized_saddle_origin(self):
        s = RegularizedSaddle()
        gx, gy = s.grad(np.array(0.0), np.array(0.0))
        assert abs(float(gx)) < 1e-10
        assert abs(float(gy)) < 1e-10

        eigvals, _ = s.hessian_eig(0.0, 0.0)
        assert eigvals[0] * eigvals[1] < 0, \
            f"Expected saddle (mixed signs), got {eigvals}"


class TestHessianEigenvalues:
    """Verify Hessian eigenvalue computation."""

    @pytest.mark.parametrize("surface", ALL_SURFACES)
    def test_hessian_symmetric(self, surface):
        """Hessian should be symmetric at random test points."""
        rng = np.random.RandomState(42)
        for _ in range(5):
            x = rng.uniform(*surface.x_range)
            y = rng.uniform(*surface.y_range)
            H = surface.hessian(x, y)
            np.testing.assert_allclose(
                H, H.T, atol=1e-6,
                err_msg=f"{surface.name}: Hessian not symmetric at ({x:.2f}, {y:.2f})",
            )


class TestSurfaceGradients:
    """Finite-difference gradient checks for surfaces."""

    @pytest.mark.parametrize("surface", ALL_SURFACES)
    def test_gradient_finite_diff(self, surface):
        """Analytical gradient should match FD at test points."""
        eps = 1e-5
        rng = np.random.RandomState(42)

        for _ in range(5):
            x = rng.uniform(*surface.x_range) * 0.5  # stay near center
            y = rng.uniform(*surface.y_range) * 0.5

            gx_a, gy_a = surface.grad(np.array(x), np.array(y))
            gx_fd = (surface.f_scalar(x + eps, y) - surface.f_scalar(x - eps, y)) / (2 * eps)
            gy_fd = (surface.f_scalar(x, y + eps) - surface.f_scalar(x, y - eps)) / (2 * eps)

            np.testing.assert_allclose(
                float(gx_a), gx_fd, atol=1e-3, rtol=1e-3,
                err_msg=f"{surface.name}: dx mismatch at ({x:.3f}, {y:.3f})",
            )
            np.testing.assert_allclose(
                float(gy_a), gy_fd, atol=1e-3, rtol=1e-3,
                err_msg=f"{surface.name}: dy mismatch at ({x:.3f}, {y:.3f})",
            )
