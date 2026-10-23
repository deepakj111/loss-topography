"""
2D Parameter Terrains — Critical Points & Pathological Topologies.

Each surface exposes f(x, y), grad(x, y), hessian(x, y), and annotated
critical points classified by Hessian eigenvalues.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class CriticalPoint:
    """A critical point on a surface where ∇f = 0."""
    coords: Tuple[float, float]
    f_val: float
    eigenvalues: Tuple[float, float]
    kind: str   # "local_min" | "global_min" | "local_max" | "saddle" | "degenerate"
    label: str = ""


@dataclass
class Surface:
    """2D test surface for optimizer visualization."""
    name: str
    latex: str
    x_range: Tuple[float, float] = (-3.0, 3.0)
    y_range: Tuple[float, float] = (-3.0, 3.0)
    critical_points: List[CriticalPoint] = field(default_factory=list)

    def f(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def hessian(self, x: float, y: float) -> np.ndarray:
        """Return 2x2 Hessian at a single point."""
        raise NotImplementedError

    def hessian_eig(self, x: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues and eigenvectors of the Hessian."""
        H = self.hessian(x, y)
        eigvals, eigvecs = np.linalg.eigh(H)
        return eigvals, eigvecs

    def meshgrid(self, n: int = 200):
        xs = np.linspace(*self.x_range, n)
        ys = np.linspace(*self.y_range, n)
        X, Y = np.meshgrid(xs, ys)
        Z = self.f(X, Y)
        return X, Y, Z

    def f_scalar(self, x: float, y: float) -> float:
        return float(self.f(np.array(x), np.array(y)))


# ── 1. Strict Saddle  f(x,y) = x² − y² ──────────────────────────

class StrictSaddle(Surface):
    def __init__(self):
        super().__init__(
            name="Strict Saddle",
            latex=r"$f(x,y) = x^2 - y^2$",
            x_range=(-2.5, 2.5),
            y_range=(-2.5, 2.5),
            critical_points=[
                CriticalPoint((0, 0), 0.0, (-2.0, 2.0), "saddle", "Origin Saddle"),
            ],
        )

    def f(self, x, y):
        return x ** 2 - y ** 2

    def grad(self, x, y):
        return 2 * x, -2 * y

    def hessian(self, x, y):
        return np.array([[2.0, 0.0], [0.0, -2.0]])


# ── 2. Monkey Saddle  f(x,y) = x³ − 3xy² ────────────────────────

class MonkeySaddle(Surface):
    def __init__(self):
        super().__init__(
            name="Monkey Saddle",
            latex=r"$f(x,y) = x^3 - 3xy^2$",
            x_range=(-2, 2),
            y_range=(-2, 2),
            critical_points=[
                CriticalPoint((0, 0), 0.0, (0.0, 0.0), "degenerate",
                              "Degenerate (H=0)"),
            ],
        )

    def f(self, x, y):
        return x ** 3 - 3 * x * y ** 2

    def grad(self, x, y):
        return 3 * x ** 2 - 3 * y ** 2, -6 * x * y

    def hessian(self, x, y):
        return np.array([[6 * x, -6 * y], [-6 * y, -6 * x]])


# ── 3. Rosenbrock Ravine ─────────────────────────────────────────

class Rosenbrock(Surface):
    def __init__(self, a: float = 1.0, b: float = 100.0):
        self.a, self.b = a, b
        super().__init__(
            name="Rosenbrock",
            latex=r"$f=(a-x)^2 + b(y-x^2)^2$",
            x_range=(-2.0, 2.0),
            y_range=(-1.0, 3.0),
            critical_points=[
                CriticalPoint((a, a ** 2), 0.0, (2.0, 200.0), "global_min",
                              f"Global min ({a},{a**2})"),
            ],
        )

    def f(self, x, y):
        return (self.a - x) ** 2 + self.b * (y - x ** 2) ** 2

    def grad(self, x, y):
        dx = -2 * (self.a - x) - 4 * self.b * x * (y - x ** 2)
        dy = 2 * self.b * (y - x ** 2)
        return dx, dy

    def hessian(self, x, y):
        a, b = self.a, self.b
        dxx = 2 + 8 * b * x ** 2 - 4 * b * (y - x ** 2)
        dxy = -4 * b * x
        dyy = 2 * b
        return np.array([[dxx, dxy], [dxy, dyy]])


# ── 4. Rastrigin ─────────────────────────────────────────────────

class Rastrigin(Surface):
    def __init__(self, A: float = 10.0):
        self.A = A
        cps = [CriticalPoint((0, 0), 0.0, (2*A*(1 + 2*np.pi**2), 2*A*(1 + 2*np.pi**2)),
                              "global_min", "Global min")]
        super().__init__(
            name="Rastrigin",
            latex=r"$2A + x^2 - A\cos(2\pi x) + y^2 - A\cos(2\pi y)$",
            x_range=(-3, 3),
            y_range=(-3, 3),
            critical_points=cps,
        )

    def f(self, x, y):
        A = self.A
        return 2 * A + (x ** 2 - A * np.cos(2 * np.pi * x)) + \
               (y ** 2 - A * np.cos(2 * np.pi * y))

    def grad(self, x, y):
        A = self.A
        dx = 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
        dy = 2 * y + 2 * np.pi * A * np.sin(2 * np.pi * y)
        return dx, dy

    def hessian(self, x, y):
        A = self.A
        dxx = 2 + 4 * np.pi ** 2 * A * np.cos(2 * np.pi * x)
        dyy = 2 + 4 * np.pi ** 2 * A * np.cos(2 * np.pi * y)
        return np.array([[dxx, 0.0], [0.0, dyy]])


# ── 5. Styblinski-Tang ──────────────────────────────────────────

class StyblinskiTang(Surface):
    def __init__(self):
        super().__init__(
            name="Styblinski-Tang",
            latex=r"$\frac{1}{2}\sum(x_i^4 - 16x_i^2 + 5x_i)$",
            x_range=(-5, 5),
            y_range=(-5, 5),
            critical_points=[
                CriticalPoint((-2.9035, -2.9035), -78.332, (4.0, 4.0),
                              "global_min", "Global min"),
            ],
        )

    def f(self, x, y):
        return 0.5 * ((x ** 4 - 16 * x ** 2 + 5 * x) +
                       (y ** 4 - 16 * y ** 2 + 5 * y))

    def grad(self, x, y):
        dx = 0.5 * (4 * x ** 3 - 32 * x + 5)
        dy = 0.5 * (4 * y ** 3 - 32 * y + 5)
        return dx, dy

    def hessian(self, x, y):
        dxx = 0.5 * (12 * x ** 2 - 32)
        dyy = 0.5 * (12 * y ** 2 - 32)
        return np.array([[dxx, 0.0], [0.0, dyy]])


# ── 6. Beale Function ────────────────────────────────────────────

class Beale(Surface):
    def __init__(self):
        super().__init__(
            name="Beale",
            latex=r"$(1.5-x+xy)^2+(2.25-x+xy^2)^2+(2.625-x+xy^3)^2$",
            x_range=(-4.5, 4.5),
            y_range=(-4.5, 4.5),
            critical_points=[
                CriticalPoint((3.0, 0.5), 0.0, (0.5, 14.2), "global_min",
                              "Global min (3, 0.5)"),
            ],
        )

    def f(self, x, y):
        return ((1.5 - x + x * y) ** 2 +
                (2.25 - x + x * y ** 2) ** 2 +
                (2.625 - x + x * y ** 3) ** 2)

    def grad(self, x, y):
        a = 1.5 - x + x * y
        b = 2.25 - x + x * y ** 2
        c = 2.625 - x + x * y ** 3
        dx = 2 * a * (-1 + y) + 2 * b * (-1 + y ** 2) + 2 * c * (-1 + y ** 3)
        dy = 2 * a * x + 2 * b * (2 * x * y) + 2 * c * (3 * x * y ** 2)
        return dx, dy

    def hessian(self, x, y):
        eps = 1e-5
        gx0, gy0 = self.grad(x, y)
        gx1, _ = self.grad(x + eps, y)
        _, gy1 = self.grad(x, y + eps)
        gx2, gy2 = self.grad(x + eps, y + eps)
        dxx = (gx1 - gx0) / eps
        dyy = (gy1 - gy0) / eps
        gx_dy, _ = self.grad(x, y + eps)
        dxy = (gx_dy - gx0) / eps
        return np.array([[dxx, dxy], [dxy, dyy]])


# ── 7. Saddle with quartic regularization ────────────────────────
#    Used for saddle escape animation

class RegularizedSaddle(Surface):
    """f(x,y) = x² − y² + 0.1x⁴ + 0.1y⁴  (bounded saddle)."""

    def __init__(self):
        super().__init__(
            name="Regularized Saddle",
            latex=r"$x^2 - y^2 + 0.1x^4 + 0.1y^4$",
            x_range=(-2.5, 2.5),
            y_range=(-2.5, 2.5),
            critical_points=[
                CriticalPoint((0, 0), 0.0, (-2.0, 2.0), "saddle",
                              "Saddle at origin"),
            ],
        )

    def f(self, x, y):
        return x ** 2 - y ** 2 + 0.1 * x ** 4 + 0.1 * y ** 4

    def grad(self, x, y):
        return 2 * x + 0.4 * x ** 3, -2 * y + 0.4 * y ** 3

    def hessian(self, x, y):
        return np.array([
            [2.0 + 1.2 * x ** 2, 0.0],
            [0.0, -2.0 + 1.2 * y ** 2],
        ])


# ── 8. Twin-Well Sharp vs Flat Minimum ──────────────────────────

class TwinWell(Surface):
    """
    Custom analytical potential with two global minima at the same loss height:
    - Sharp minimum at x ≈ -2  (Tr(H) ≈ 80, narrow canyon)
    - Flat minimum  at x ≈ +2  (Tr(H) ≈ 2, wide basin)
    """

    def __init__(self):
        super().__init__(
            name="Twin Well (Sharp vs Flat)",
            latex=r"Sharp $\mathrm{Tr}(\mathbf{H})=80$ vs Flat $\mathrm{Tr}(\mathbf{H})=2$",
            x_range=(-4, 4),
            y_range=(-3, 3),
        )
        # Precompute critical points
        self.critical_points = [
            CriticalPoint((-2, 0), 0.0, (40.0, 40.0), "global_min",
                          "Sharp min (Tr=80)"),
            CriticalPoint((2, 0), 0.0, (1.0, 1.0), "global_min",
                          "Flat min (Tr=2)"),
            CriticalPoint((0, 0), self.f_scalar(0, 0),
                          (-1.0, 1.0), "saddle", "Saddle barrier"),
        ]

    def f(self, x, y):
        # Sharp well: 20 * ((x+2)² + y²) * exp(- ((x+2)² + y²))
        # Flat well:  0.5 * ((x-2)² + y²) * exp(-0.1*((x-2)² + y²))
        # Plus a global offset so both minima touch 0
        sharp_r2 = (x + 2) ** 2 + y ** 2
        flat_r2 = (x - 2) ** 2 + y ** 2

        # Mexican-hat style wells of different curvature
        sharp = 20.0 * sharp_r2 * np.exp(-2.0 * sharp_r2)
        flat = 0.5 * flat_r2 * np.exp(-0.05 * flat_r2)

        # Barrier: gentle bump at center
        barrier = 3.0 * np.exp(-0.5 * (x ** 2 + y ** 2))

        # Invert and shift so minima are at 0
        return -sharp - flat + barrier + 3.0

    def grad(self, x, y):
        eps = 1e-6
        fx = (self.f(x + eps, y) - self.f(x - eps, y)) / (2 * eps)
        fy = (self.f(x, y + eps) - self.f(x, y - eps)) / (2 * eps)
        return fx, fy

    def hessian(self, x, y):
        eps = 1e-4
        f = self.f
        fxx = (f(x + eps, y) - 2 * f(x, y) + f(x - eps, y)) / eps ** 2
        fyy = (f(x, y + eps) - 2 * f(x, y) + f(x, y - eps)) / eps ** 2
        fxy = (f(x + eps, y + eps) - f(x + eps, y - eps) -
               f(x - eps, y + eps) + f(x - eps, y - eps)) / (4 * eps ** 2)
        return np.array([[float(fxx), float(fxy)], [float(fxy), float(fyy)]])


# ── 9. Multi-Modal Arena ─────────────────────────────────────────

class MultiModal(Surface):
    """
    Combined terrain with 1 deep global min, 3 local minima traps,
    and interconnecting saddle passes.
    """

    def __init__(self):
        super().__init__(
            name="Multi-Modal Arena",
            latex=r"Custom multi-modal landscape",
            x_range=(-4, 4),
            y_range=(-4, 4),
        )
        self.critical_points = [
            CriticalPoint((0, 0), -5.0, (2.0, 2.0), "global_min",
                          "Global min"),
            CriticalPoint((-2.5, 2.5), -2.0, (1.0, 1.0), "local_min",
                          "Local trap 1"),
            CriticalPoint((2.5, -2.5), -2.0, (1.0, 1.0), "local_min",
                          "Local trap 2"),
            CriticalPoint((-2.5, -2.5), -1.5, (1.0, 1.0), "local_min",
                          "Local trap 3"),
        ]

    def f(self, x, y):
        # Deep global well at origin
        z = -5.0 * np.exp(-0.5 * (x ** 2 + y ** 2))
        # Three local minima traps
        z -= 2.0 * np.exp(-0.8 * ((x + 2.5) ** 2 + (y - 2.5) ** 2))
        z -= 2.0 * np.exp(-0.8 * ((x - 2.5) ** 2 + (y + 2.5) ** 2))
        z -= 1.5 * np.exp(-0.8 * ((x + 2.5) ** 2 + (y + 2.5) ** 2))
        # Raised base to create ridges and saddles
        z += 0.05 * (x ** 2 + y ** 2)
        # Rastrigin-like ripples for extra deceptiveness
        z += 0.3 * (np.cos(2 * np.pi * x * 0.5) + np.cos(2 * np.pi * y * 0.5))
        return z

    def grad(self, x, y):
        eps = 1e-6
        fx = (self.f(x + eps, y) - self.f(x - eps, y)) / (2 * eps)
        fy = (self.f(x, y + eps) - self.f(x, y - eps)) / (2 * eps)
        return fx, fy

    def hessian(self, x, y):
        eps = 1e-4
        f = self.f
        fxx = (f(x + eps, y) - 2 * f(x, y) + f(x - eps, y)) / eps ** 2
        fyy = (f(x, y + eps) - 2 * f(x, y) + f(x, y - eps)) / eps ** 2
        fxy = (f(x + eps, y + eps) - f(x + eps, y - eps) -
               f(x - eps, y + eps) + f(x - eps, y - eps)) / (4 * eps ** 2)
        return np.array([[float(fxx), float(fxy)], [float(fxy), float(fyy)]])


# ── Registry ─────────────────────────────────────────────────────

ALL_SURFACES = [
    StrictSaddle(), MonkeySaddle(), Rosenbrock(), Rastrigin(),
    StyblinskiTang(), Beale(), RegularizedSaddle(), TwinWell(),
    MultiModal(),
]

SURFACE_REGISTRY = {s.name: s for s in ALL_SURFACES}
