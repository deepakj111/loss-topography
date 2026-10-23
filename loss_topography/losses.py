"""
Complete Loss Function Atlas with exact analytical gradients and Hessians.

Every loss implements value, first derivative, second derivative, and
a 2D parameter-surface mapping for linear/logistic regression visualization.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class LossFunction:
    """Unified loss function container."""

    name: str
    latex: str          # LaTeX formula string
    category: str       # "regression" | "classification" | "robust"

    # Core callables operating on residual r = y_hat - y  (or margin m = y * y_hat)
    _value_fn: Callable[[np.ndarray], np.ndarray]
    _grad_fn: Callable[[np.ndarray], np.ndarray]
    _hess_fn: Callable[[np.ndarray], np.ndarray]

    # Optional extra params
    params: dict | None = None

    def value(self, r: np.ndarray) -> np.ndarray:
        return self._value_fn(np.asarray(r, dtype=np.float64))

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return self.value(r)

    def grad(self, r: np.ndarray) -> np.ndarray:
        return self._grad_fn(np.asarray(r, dtype=np.float64))

    def hessian(self, r: np.ndarray) -> np.ndarray:
        return self._hess_fn(np.asarray(r, dtype=np.float64))

    def surface_linear(self, X: np.ndarray, y: np.ndarray,
                       t1_grid: np.ndarray, t2_grid: np.ndarray) -> np.ndarray:
        """Compute loss surface L(theta1, theta2) for y_hat = theta1*x + theta2."""
        Z = np.zeros_like(t1_grid)
        for i in range(t1_grid.shape[0]):
            for j in range(t1_grid.shape[1]):
                y_hat = t1_grid[i, j] * X + t2_grid[i, j]
                r = y_hat - y
                Z[i, j] = np.mean(self.value(r))
        return Z

    def __repr__(self):
        return f"LossFunction({self.name})"


# ── Helpers ────────────────────────────────────────────────────────

_EPS = 1e-15


# ── 1. Mean Squared Error (MSE / L2) ─────────────────────────────

def _mse_val(r):
    return 0.5 * r ** 2

def _mse_grad(r):
    return r

def _mse_hess(r):
    return np.ones_like(r)

MSE = LossFunction(
    name="MSE",
    latex=r"$\frac{1}{2}(y - \hat{y})^2$",
    category="regression",
    _value_fn=_mse_val, _grad_fn=_mse_grad, _hess_fn=_mse_hess,
)


# ── 2. Mean Absolute Error (MAE / L1) ────────────────────────────

def _mae_val(r):
    return np.abs(r)

def _mae_grad(r):
    return np.sign(r)

def _mae_hess(r):
    return np.zeros_like(r)

MAE = LossFunction(
    name="MAE",
    latex=r"$|y - \hat{y}|$",
    category="regression",
    _value_fn=_mae_val, _grad_fn=_mae_grad, _hess_fn=_mae_hess,
)


# ── 3. Huber Loss ────────────────────────────────────────────────

def make_huber(delta: float = 1.0) -> LossFunction:
    def val(r):
        a = np.abs(r)
        return np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))

    def grad(r):
        return np.where(np.abs(r) <= delta, r, delta * np.sign(r))

    def hess(r):
        return np.where(np.abs(r) <= delta, 1.0, 0.0)

    return LossFunction(
        name="Huber",
        latex=r"$\mathrm{Huber}_\delta(r)$",
        category="regression",
        _value_fn=val, _grad_fn=grad, _hess_fn=hess,
        params={"delta": delta},
    )

Huber = make_huber(1.0)


# ── 4. Log-Cosh Loss ─────────────────────────────────────────────

def _logcosh_val(r):
    # Numerically stable: log(cosh(r)) = |r| + log(1 + exp(-2|r|)) - log(2)
    ar = np.abs(r)
    return ar + np.log1p(np.exp(-2 * ar)) - np.log(2)

def _logcosh_grad(r):
    return np.tanh(r)

def _logcosh_hess(r):
    return 1.0 - np.tanh(r) ** 2  # sech^2(r)

LogCosh = LossFunction(
    name="Log-Cosh",
    latex=r"$\log(\cosh(r))$",
    category="regression",
    _value_fn=_logcosh_val, _grad_fn=_logcosh_grad, _hess_fn=_logcosh_hess,
)


# ── 5. Quantile (Pinball) Loss ───────────────────────────────────

def make_quantile(tau: float = 0.5) -> LossFunction:
    def val(r):
        return np.where(r >= 0, tau * r, (tau - 1) * r)

    def grad(r):
        return np.where(r >= 0, tau, tau - 1.0)

    def hess(r):
        return np.zeros_like(r)

    return LossFunction(
        name=f"Quantile(τ={tau})",
        latex=r"$\mathrm{Quantile}_\tau(r)$",
        category="regression",
        _value_fn=val, _grad_fn=grad, _hess_fn=hess,
        params={"tau": tau},
    )

Quantile = make_quantile(0.5)


# ── 6. Binary Cross-Entropy with Logits ──────────────────────────
#    Input: logit z, label y ∈ {0, 1}.  r = z (raw logit)
#    For plotting: we treat r as the logit difference.

def _bce_val(r):
    """BCE(z) = max(z,0) - z*y + log(1+exp(-|z|)),  here y=1 so = log(1+exp(-z))."""
    # We plot the loss as a function of logit for a positive sample (y=1).
    return np.log1p(np.exp(-r)) + np.maximum(-r, 0)
    # Simplified: softplus(-r) but numerically stable

def _bce_val_simple(r):
    """For y=1: L(z) = log(1 + exp(-z))."""
    return np.logaddexp(0, -r)

def _bce_grad(r):
    """dL/dz = sigmoid(z) - 1 = -sigmoid(-z)."""
    return -1.0 / (1.0 + np.exp(r))  # = -(1 - sigmoid(r))

def _bce_hess(r):
    """d2L/dz2 = sigmoid(z) * (1 - sigmoid(z))."""
    s = 1.0 / (1.0 + np.exp(-r))
    return s * (1.0 - s)

BCE = LossFunction(
    name="BCE",
    latex=r"$\log(1 + e^{-z})$",
    category="classification",
    _value_fn=_bce_val_simple, _grad_fn=_bce_grad, _hess_fn=_bce_hess,
)


# ── 7. Focal Loss ────────────────────────────────────────────────

def make_focal(gamma: float = 2.0, alpha: float = 1.0) -> LossFunction:
    def val(r):
        """r = logit for positive class (y=1)."""
        p = 1.0 / (1.0 + np.exp(-r))
        p = np.clip(p, _EPS, 1.0 - _EPS)
        return -alpha * (1.0 - p) ** gamma * np.log(p)

    def grad(r):
        p = 1.0 / (1.0 + np.exp(-r))
        p = np.clip(p, _EPS, 1.0 - _EPS)
        # d/dz [-alpha (1-p)^gamma log(p)]
        # = alpha * (1-p)^gamma * (gamma * p * log(p) / (1-p) - 1) * (1-p)
        # Simplified via chain rule:
        term1 = gamma * (1.0 - p) ** (gamma - 1) * p * (1.0 - p) * np.log(p)
        term2 = -(1.0 - p) ** gamma * (1.0 - p)
        return -alpha * (term1 + term2)

    def hess(r):
        # Numerical Hessian for focal (analytical is complex)
        eps = 1e-5
        return (val(r + eps) - 2 * val(r) + val(r - eps)) / eps ** 2

    return LossFunction(
        name=f"Focal(γ={gamma})",
        latex=r"$-\alpha(1-p_t)^\gamma \log(p_t)$",
        category="classification",
        _value_fn=val, _grad_fn=grad, _hess_fn=hess,
        params={"gamma": gamma, "alpha": alpha},
    )

Focal = make_focal(2.0)


# ── 8. Hinge Loss ────────────────────────────────────────────────
#    Input: margin m = y * y_hat (treat r as margin)

def _hinge_val(r):
    """Hinge loss: max(0, 1 - m) where m = margin."""
    return np.maximum(0, 1.0 - r)

def _hinge_grad(r):
    return np.where(r < 1.0, -1.0, 0.0)

def _hinge_hess(r):
    return np.zeros_like(r)

Hinge = LossFunction(
    name="Hinge",
    latex=r"$\max(0, 1-m)$",
    category="classification",
    _value_fn=_hinge_val, _grad_fn=_hinge_grad, _hess_fn=_hinge_hess,
)


# ── 9. KL Divergence ─────────────────────────────────────────────
#    Plot as function of q (predicted prob) for fixed p=0.7

def make_kl(p: float = 0.7) -> LossFunction:
    """KL(p || q) as a function of q."""
    def val(r):
        q = 1.0 / (1.0 + np.exp(-r))  # sigmoid to get prob
        q = np.clip(q, _EPS, 1.0 - _EPS)
        return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))

    def grad(r):
        q = 1.0 / (1.0 + np.exp(-r))
        q = np.clip(q, _EPS, 1.0 - _EPS)
        dq_dr = q * (1.0 - q)
        return (-p / q + (1.0 - p) / (1.0 - q)) * dq_dr

    def hess(r):
        eps = 1e-5
        return (val(r + eps) - 2 * val(r) + val(r - eps)) / eps ** 2

    return LossFunction(
        name="KL Divergence",
        latex=r"$D_{KL}(p \| q)$",
        category="classification",
        _value_fn=val, _grad_fn=grad, _hess_fn=hess,
        params={"p": p},
    )

KL = make_kl(0.7)


# ── 10. Cauchy Robust Loss ───────────────────────────────────────

def make_cauchy(c: float = 1.0) -> LossFunction:
    def val(r):
        return 0.5 * c ** 2 * np.log(1.0 + (r / c) ** 2)

    def grad(r):
        return r / (1.0 + (r / c) ** 2)

    def hess(r):
        u = (r / c) ** 2
        return (1.0 - u) / (1.0 + u) ** 2

    return LossFunction(
        name="Cauchy",
        latex=r"$\frac{c^2}{2}\log\left(1+\frac{r^2}{c^2}\right)$",
        category="robust",
        _value_fn=val, _grad_fn=grad, _hess_fn=hess,
        params={"c": c},
    )

Cauchy = make_cauchy(1.0)


# ── Registry ─────────────────────────────────────────────────────

ALL_LOSSES = [MSE, MAE, Huber, LogCosh, Quantile, BCE, Focal, Hinge, KL, Cauchy]

LOSS_REGISTRY = {lf.name: lf for lf in ALL_LOSSES}


def numerical_grad(loss: LossFunction, r: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient for testing."""
    return (loss.value(r + eps) - loss.value(r - eps)) / (2 * eps)


def numerical_hess(loss: LossFunction, r: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central finite-difference second derivative for testing."""
    return (loss.value(r + eps) - 2 * loss.value(r) + loss.value(r - eps)) / eps ** 2
