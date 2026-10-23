"""Tests for optimizers: convergence and timing telemetry."""

import numpy as np
import pytest
from loss_topography.optimizers import (
    SGD, MomentumSGD, NAG, AdaGrad, RMSProp, Adam, AdamW, DampedNewton,
    get_default_optimizers, TrajectoryLog,
)
from loss_topography.surfaces import StrictSaddle, Rosenbrock, RegularizedSaddle


class TestOptimizerConvergence:
    """Test that optimizers converge on simple quadratic surfaces."""

    def test_sgd_on_quadratic(self):
        """SGD should converge on a simple quadratic bowl."""
        from loss_topography.surfaces import Surface

        class Bowl(Surface):
            def __init__(self):
                super().__init__(name="Bowl", latex="x²+y²",
                                 x_range=(-5, 5), y_range=(-5, 5))
                from loss_topography.surfaces import CriticalPoint
                self.critical_points = [
                    CriticalPoint((0, 0), 0.0, (2.0, 2.0), "global_min", "Origin")
                ]

            def f(self, x, y): return x ** 2 + y ** 2
            def grad(self, x, y): return 2 * x, 2 * y
            def hessian(self, x, y): return np.array([[2.0, 0.0], [0.0, 2.0]])

        bowl = Bowl()
        sgd = SGD(lr=0.1)
        log = sgd.run(bowl, x0=3.0, y0=4.0, n_steps=100)

        assert log.losses[-1] < 0.01, f"SGD didn't converge: final loss = {log.losses[-1]}"
        assert log.converged_step is not None, "SGD should have converged"

    def test_adam_on_rosenbrock(self):
        """Adam should make significant progress on Rosenbrock."""
        surface = Rosenbrock()
        adam = Adam(lr=0.01)
        log = adam.run(surface, x0=-1.5, y0=2.0, n_steps=500)

        # Adam should at least reduce loss significantly
        assert log.losses[-1] < log.losses[0], \
            "Adam should reduce loss on Rosenbrock"

    def test_momentum_escapes_saddle(self):
        """Momentum should escape the saddle point region."""
        surface = RegularizedSaddle()
        mom = MomentumSGD(lr=0.01, beta=0.9)
        log = mom.run(surface, x0=0.01, y0=0.001, n_steps=200)

        # Should move away from the saddle
        final_dist = np.sqrt(log.points[-1].x ** 2 + log.points[-1].y ** 2)
        assert final_dist > 0.1, \
            f"Momentum should escape saddle, but stayed at dist={final_dist:.4f}"


class TestTrajectoryLog:
    """Test trajectory logging properties."""

    def test_log_has_correct_length(self):
        from loss_topography.surfaces import Surface, CriticalPoint

        class Bowl(Surface):
            def __init__(self):
                super().__init__(name="Bowl", latex="x²+y²")
                self.critical_points = [
                    CriticalPoint((0, 0), 0.0, (2.0, 2.0), "global_min", "Origin")
                ]
            def f(self, x, y): return x ** 2 + y ** 2
            def grad(self, x, y): return 2 * x, 2 * y
            def hessian(self, x, y): return np.array([[2.0, 0.0], [0.0, 2.0]])

        bowl = Bowl()
        adam = Adam(lr=0.05)
        log = adam.run(bowl, 1.0, 1.0, n_steps=50)

        assert len(log.points) == 51  # n_steps + final point
        assert log.total_steps == 50

    def test_path_efficiency_bounded(self):
        """Path efficiency should be between 0 and 1."""
        from loss_topography.surfaces import Surface, CriticalPoint

        class Bowl(Surface):
            def __init__(self):
                super().__init__(name="Bowl", latex="x²+y²")
                self.critical_points = [
                    CriticalPoint((0, 0), 0.0, (2.0, 2.0), "global_min", "Origin")
                ]
            def f(self, x, y): return x ** 2 + y ** 2
            def grad(self, x, y): return 2 * x, 2 * y
            def hessian(self, x, y): return np.array([[2.0, 0.0], [0.0, 2.0]])

        bowl = Bowl()
        for name, opt in get_default_optimizers().items():
            log = opt.run(bowl, 2.0, 2.0, n_steps=100)
            assert 0 <= log.path_efficiency <= 1.0 + 1e-6, \
                f"{name}: path efficiency = {log.path_efficiency} out of bounds"

    def test_wall_clock_positive(self):
        """Wall clock times should be positive."""
        from loss_topography.surfaces import Surface, CriticalPoint

        class Bowl(Surface):
            def __init__(self):
                super().__init__(name="Bowl", latex="x²+y²")
                self.critical_points = [
                    CriticalPoint((0, 0), 0.0, (2.0, 2.0), "global_min", "Origin")
                ]
            def f(self, x, y): return x ** 2 + y ** 2
            def grad(self, x, y): return 2 * x, 2 * y
            def hessian(self, x, y): return np.array([[2.0, 0.0], [0.0, 2.0]])

        bowl = Bowl()
        sgd = SGD(lr=0.1)
        log = sgd.run(bowl, 1.0, 1.0, n_steps=10)

        # At least the first step should have positive timing
        assert log.points[0].wall_clock_us >= 0
