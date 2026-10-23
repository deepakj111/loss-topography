"""
From-scratch NumPy optimizers with full trajectory logging and timing.

Implements SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW, and Damped Newton.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional


@dataclass
class TrajectoryPoint:
    """Single point in an optimizer trajectory."""
    x: float
    y: float
    loss: float
    grad_norm: float
    eigenvalues: Tuple[float, float]
    condition_number: float
    wall_clock_us: float  # microseconds for this step


@dataclass
class TrajectoryLog:
    """Full trajectory from an optimizer run."""
    optimizer_name: str
    points: List[TrajectoryPoint] = field(default_factory=list)
    total_steps: int = 0
    converged_step: Optional[int] = None  # step where ε-convergence reached
    path_efficiency: float = 0.0  # straight-line / arc-length ratio

    @property
    def xs(self) -> np.ndarray:
        return np.array([p.x for p in self.points])

    @property
    def ys(self) -> np.ndarray:
        return np.array([p.y for p in self.points])

    @property
    def losses(self) -> np.ndarray:
        return np.array([p.loss for p in self.points])

    @property
    def grad_norms(self) -> np.ndarray:
        return np.array([p.grad_norm for p in self.points])

    @property
    def wall_clocks(self) -> np.ndarray:
        return np.array([p.wall_clock_us for p in self.points])


def _compute_path_efficiency(points: List[TrajectoryPoint]) -> float:
    """Euclidean(start→end) / total arc length."""
    if len(points) < 2:
        return 1.0
    straight = np.sqrt((points[-1].x - points[0].x) ** 2 +
                        (points[-1].y - points[0].y) ** 2)
    arc = sum(
        np.sqrt((points[i + 1].x - points[i].x) ** 2 +
                (points[i + 1].y - points[i].y) ** 2)
        for i in range(len(points) - 1)
    )
    if arc < 1e-12:
        return 1.0
    return straight / arc


class Optimizer:
    """Base class for all optimizers."""

    name: str = "Base"

    def step(self, x: float, y: float,
             grad_fn: Callable, hessian_fn: Callable = None) -> Tuple[float, float]:
        raise NotImplementedError

    def reset(self):
        """Reset internal state for a fresh run."""
        pass

    def run(self, surface, x0: float, y0: float,
            n_steps: int = 200, eps: float = 1e-3,
            f_star: Optional[float] = None) -> TrajectoryLog:
        """Run the optimizer on a surface and record full trajectory."""
        self.reset()
        log = TrajectoryLog(optimizer_name=self.name)

        x, y = x0, y0

        # Determine global minimum value
        if f_star is None:
            f_star_val = min(
                (cp.f_val for cp in surface.critical_points
                 if cp.kind in ("global_min", "local_min")),
                default=0.0,
            )
        else:
            f_star_val = f_star

        for step_idx in range(n_steps):
            t0 = time.perf_counter_ns()

            loss_val = surface.f_scalar(x, y)
            gx, gy = surface.grad(np.array(x), np.array(y))
            gx, gy = float(gx), float(gy)

            # Protect against overflow
            gx = np.clip(gx, -1e15, 1e15)
            gy = np.clip(gy, -1e15, 1e15)
            grad_norm = np.sqrt(gx ** 2 + gy ** 2)
            if not np.isfinite(loss_val):
                loss_val = 1e15

            try:
                H = surface.hessian(x, y)
                eigvals = np.linalg.eigvalsh(H)
                cond = abs(eigvals[-1] / eigvals[0]) if abs(eigvals[0]) > 1e-12 else float('inf')
            except Exception:
                eigvals = np.array([0.0, 0.0])
                cond = 1.0

            # Step
            x_new, y_new = self.step(x, y,
                                     grad_fn=lambda xx, yy: surface.grad(np.array(xx), np.array(yy)),
                                     hessian_fn=lambda xx, yy: surface.hessian(xx, yy))

            # Clip to prevent divergence to infinity while preventing floating-point overflow
            x_new = np.clip(x_new, -50.0, 50.0)
            y_new = np.clip(y_new, -50.0, 50.0)
            if not (np.isfinite(x_new) and np.isfinite(y_new)):
                x_new, y_new = x, y  # stay put if step diverged

            dt_us = (time.perf_counter_ns() - t0) / 1e3

            pt = TrajectoryPoint(
                x=x, y=y, loss=loss_val, grad_norm=grad_norm,
                eigenvalues=(float(eigvals[0]), float(eigvals[-1])),
                condition_number=cond, wall_clock_us=dt_us,
            )
            log.points.append(pt)

            # Check convergence
            if log.converged_step is None and abs(loss_val - f_star_val) < eps:
                log.converged_step = step_idx

            x, y = x_new, y_new

        # Final point
        loss_val = surface.f_scalar(x, y)
        log.points.append(TrajectoryPoint(
            x=x, y=y, loss=loss_val, grad_norm=0.0,
            eigenvalues=(0.0, 0.0), condition_number=1.0, wall_clock_us=0.0,
        ))

        log.total_steps = n_steps
        log.path_efficiency = _compute_path_efficiency(log.points)
        return log


# ── SGD ──────────────────────────────────────────────────────────

class SGD(Optimizer):
    name = "SGD"

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, x, y, grad_fn, hessian_fn=None):
        gx, gy = grad_fn(x, y)
        return x - self.lr * float(gx), y - self.lr * float(gy)

    def reset(self):
        pass


# ── SGD + Momentum ───────────────────────────────────────────────

class MomentumSGD(Optimizer):
    name = "Momentum"

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.vx = 0.0
        self.vy = 0.0

    def step(self, x, y, grad_fn, hessian_fn=None):
        gx, gy = grad_fn(x, y)
        gx, gy = float(gx), float(gy)
        self.vx = self.beta * self.vx + gx
        self.vy = self.beta * self.vy + gy
        return x - self.lr * self.vx, y - self.lr * self.vy

    def reset(self):
        self.vx = self.vy = 0.0


# ── Nesterov Accelerated Gradient ────────────────────────────────

class NAG(Optimizer):
    name = "NAG"

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.vx = 0.0
        self.vy = 0.0

    def step(self, x, y, grad_fn, hessian_fn=None):
        # Look-ahead
        x_ahead = x - self.beta * self.lr * self.vx
        y_ahead = y - self.beta * self.lr * self.vy
        gx, gy = grad_fn(x_ahead, y_ahead)
        gx, gy = float(gx), float(gy)
        self.vx = self.beta * self.vx + gx
        self.vy = self.beta * self.vy + gy
        return x - self.lr * self.vx, y - self.lr * self.vy

    def reset(self):
        self.vx = self.vy = 0.0


# ── AdaGrad ──────────────────────────────────────────────────────

class AdaGrad(Optimizer):
    name = "AdaGrad"

    def __init__(self, lr: float = 0.5, eps: float = 1e-8):
        self.lr = lr
        self.eps = eps
        self.gx_sum = 0.0
        self.gy_sum = 0.0

    def step(self, x, y, grad_fn, hessian_fn=None):
        gx, gy = grad_fn(x, y)
        gx, gy = float(gx), float(gy)
        self.gx_sum += gx ** 2
        self.gy_sum += gy ** 2
        x_new = x - self.lr * gx / (np.sqrt(self.gx_sum) + self.eps)
        y_new = y - self.lr * gy / (np.sqrt(self.gy_sum) + self.eps)
        return x_new, y_new

    def reset(self):
        self.gx_sum = self.gy_sum = 0.0


# ── RMSProp ──────────────────────────────────────────────────────

class RMSProp(Optimizer):
    name = "RMSProp"

    def __init__(self, lr: float = 0.01, beta: float = 0.99, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.sx = 0.0
        self.sy = 0.0

    def step(self, x, y, grad_fn, hessian_fn=None):
        gx, gy = grad_fn(x, y)
        gx, gy = float(gx), float(gy)
        self.sx = self.beta * self.sx + (1 - self.beta) * gx ** 2
        self.sy = self.beta * self.sy + (1 - self.beta) * gy ** 2
        x_new = x - self.lr * gx / (np.sqrt(self.sx) + self.eps)
        y_new = y - self.lr * gy / (np.sqrt(self.sy) + self.eps)
        return x_new, y_new

    def reset(self):
        self.sx = self.sy = 0.0


# ── Adam ─────────────────────────────────────────────────────────

class Adam(Optimizer):
    name = "Adam"

    def __init__(self, lr: float = 0.05, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mx = self.my = 0.0
        self.vx = self.vy = 0.0
        self.t = 0

    def step(self, x, y, grad_fn, hessian_fn=None):
        gx, gy = grad_fn(x, y)
        gx, gy = float(gx), float(gy)
        self.t += 1
        self.mx = self.beta1 * self.mx + (1 - self.beta1) * gx
        self.my = self.beta1 * self.my + (1 - self.beta1) * gy
        self.vx = self.beta2 * self.vx + (1 - self.beta2) * gx ** 2
        self.vy = self.beta2 * self.vy + (1 - self.beta2) * gy ** 2
        mxh = self.mx / (1 - self.beta1 ** self.t)
        myh = self.my / (1 - self.beta1 ** self.t)
        vxh = self.vx / (1 - self.beta2 ** self.t)
        vyh = self.vy / (1 - self.beta2 ** self.t)
        x_new = x - self.lr * mxh / (np.sqrt(vxh) + self.eps)
        y_new = y - self.lr * myh / (np.sqrt(vyh) + self.eps)
        return x_new, y_new

    def reset(self):
        self.mx = self.my = self.vx = self.vy = 0.0
        self.t = 0


# ── AdamW ────────────────────────────────────────────────────────

class AdamW(Optimizer):
    name = "AdamW"

    def __init__(self, lr: float = 0.05, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.mx = self.my = 0.0
        self.vx = self.vy = 0.0
        self.t = 0

    def step(self, x, y, grad_fn, hessian_fn=None):
        gx, gy = grad_fn(x, y)
        gx, gy = float(gx), float(gy)
        self.t += 1
        self.mx = self.beta1 * self.mx + (1 - self.beta1) * gx
        self.my = self.beta1 * self.my + (1 - self.beta1) * gy
        self.vx = self.beta2 * self.vx + (1 - self.beta2) * gx ** 2
        self.vy = self.beta2 * self.vy + (1 - self.beta2) * gy ** 2
        mxh = self.mx / (1 - self.beta1 ** self.t)
        myh = self.my / (1 - self.beta1 ** self.t)
        vxh = self.vx / (1 - self.beta2 ** self.t)
        vyh = self.vy / (1 - self.beta2 ** self.t)
        x_new = x - self.lr * (mxh / (np.sqrt(vxh) + self.eps) + self.wd * x)
        y_new = y - self.lr * (myh / (np.sqrt(vyh) + self.eps) + self.wd * y)
        return x_new, y_new

    def reset(self):
        self.mx = self.my = self.vx = self.vy = 0.0
        self.t = 0


# ── Damped Newton ────────────────────────────────────────────────

class DampedNewton(Optimizer):
    name = "Damped Newton"

    def __init__(self, lr: float = 0.5, mu: float = 1.0):
        self.lr = lr
        self.mu = mu

    def step(self, x, y, grad_fn, hessian_fn=None):
        gx, gy = grad_fn(x, y)
        gx, gy = float(gx), float(gy)
        g = np.array([gx, gy])
        if hessian_fn is not None:
            H = hessian_fn(x, y)
            H_reg = H + self.mu * np.eye(2)
            try:
                delta = np.linalg.solve(H_reg, g)
            except np.linalg.LinAlgError:
                delta = g
        else:
            delta = g
        return x - self.lr * delta[0], y - self.lr * delta[1]

    def reset(self):
        pass


# ── Registry ─────────────────────────────────────────────────────

def get_default_optimizers() -> dict:
    """Return a dict of name → Optimizer with tuned defaults."""
    return {
        "SGD":           SGD(lr=0.01),
        "Momentum":      MomentumSGD(lr=0.01, beta=0.9),
        "NAG":           NAG(lr=0.01, beta=0.9),
        "AdaGrad":       AdaGrad(lr=0.5),
        "RMSProp":       RMSProp(lr=0.01),
        "Adam":          Adam(lr=0.05),
        "AdamW":         AdamW(lr=0.05),
        "Damped Newton": DampedNewton(lr=0.5, mu=1.0),
    }


OPTIMIZER_CLASSES = {
    "SGD": SGD,
    "Momentum": MomentumSGD,
    "NAG": NAG,
    "AdaGrad": AdaGrad,
    "RMSProp": RMSProp,
    "Adam": Adam,
    "AdamW": AdamW,
    "Damped Newton": DampedNewton,
}
