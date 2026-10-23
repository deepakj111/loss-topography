"""
Filter-Normalized Real PyTorch Neural Network Landscape Visualizer.

Implements the method from Li et al. (2018),
"Visualizing the Loss Landscape of Neural Nets".
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


def _make_moons_data(n_samples: int = 500, noise: float = 0.15,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate two-moons classification dataset."""
    rng = np.random.RandomState(seed)
    n_each = n_samples // 2

    # Upper moon
    theta1 = np.linspace(0, np.pi, n_each)
    x1 = np.column_stack([np.cos(theta1), np.sin(theta1)])

    # Lower moon
    theta2 = np.linspace(0, np.pi, n_each)
    x2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])

    X = np.vstack([x1, x2]) + rng.randn(n_samples, 2) * noise
    y = np.concatenate([np.zeros(n_each), np.ones(n_each)])
    return X.astype(np.float32), y.astype(np.float32)


def train_and_visualize_landscape(
    use_skip: bool = False,
    grid_size: int = 21,
    alpha_range: float = 1.0,
    n_epochs: int = 200,
    hidden: int = 64,
    seed: int = 42,
) -> dict:
    """
    Train a small MLP (with or without skip connections) on moons data,
    then compute the filter-normalized 2D loss landscape slice.

    Returns dict with keys: alpha_grid, beta_grid, loss_surface, model_info
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        # Return dummy data if torch not available
        alpha = np.linspace(-alpha_range, alpha_range, grid_size)
        A, B = np.meshgrid(alpha, alpha)
        Z = np.exp(-0.5 * (A ** 2 + B ** 2)) * (1 if use_skip else 3)
        return {"alpha_grid": A, "beta_grid": B, "loss_surface": Z,
                "model_info": "dummy (torch not available)"}

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_np, y_np = _make_moons_data(seed=seed)
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np).unsqueeze(1)

    # Define models
    class PlainMLP(nn.Module):
        def __init__(self, h=hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, h), nn.ReLU(),
                nn.Linear(h, h), nn.ReLU(),
                nn.Linear(h, h), nn.ReLU(),
                nn.Linear(h, h), nn.ReLU(),
                nn.Linear(h, 1),
            )

        def forward(self, x):
            return self.net(x)

    class ResidualMLP(nn.Module):
        def __init__(self, h=hidden):
            super().__init__()
            self.input_proj = nn.Linear(2, h)
            self.block1 = nn.Sequential(nn.Linear(h, h), nn.ReLU())
            self.block2 = nn.Sequential(nn.Linear(h, h), nn.ReLU())
            self.block3 = nn.Sequential(nn.Linear(h, h), nn.ReLU())
            self.block4 = nn.Sequential(nn.Linear(h, h), nn.ReLU())
            self.head = nn.Linear(h, 1)

        def forward(self, x):
            x = torch.relu(self.input_proj(x))
            x = x + self.block1(x)
            x = x + self.block2(x)
            x = x + self.block3(x)
            x = x + self.block4(x)
            return self.head(x)

    model = ResidualMLP() if use_skip else PlainMLP()
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(n_epochs):
        opt.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        opt.step()

    final_loss = criterion(model(X), y).item()
    model_info = f"{'ResidualMLP' if use_skip else 'PlainMLP'} | Final loss: {final_loss:.4f}"

    # Get trained weights as flat vector
    theta_star = torch.cat([p.data.flatten() for p in model.parameters()]).clone()

    # Generate two random directions with filter normalization
    delta = torch.randn_like(theta_star)
    eta = torch.randn_like(theta_star)

    # Filter-wise normalization: normalize each layer's direction to match layer's norm
    idx = 0
    for p in model.parameters():
        n_params = p.numel()
        d_slice = delta[idx:idx + n_params].view(p.shape)
        e_slice = eta[idx:idx + n_params].view(p.shape)
        p_norm = p.data.norm()
        if p_norm > 1e-10:
            d_slice.mul_(p_norm / (d_slice.norm() + 1e-10))
            e_slice.mul_(p_norm / (e_slice.norm() + 1e-10))
        delta[idx:idx + n_params] = d_slice.flatten()
        eta[idx:idx + n_params] = e_slice.flatten()
        idx += n_params

    # Evaluate loss landscape on grid
    alphas = np.linspace(-alpha_range, alpha_range, grid_size)
    betas = np.linspace(-alpha_range, alpha_range, grid_size)
    A, B = np.meshgrid(alphas, betas)
    Z = np.zeros_like(A)

    for i in range(grid_size):
        for j in range(grid_size):
            a, b = alphas[j], betas[i]
            new_theta = theta_star + a * delta + b * eta

            # Load perturbed weights into model
            idx = 0
            for p in model.parameters():
                n_params = p.numel()
                p.data = new_theta[idx:idx + n_params].view(p.shape)
                idx += n_params

            with torch.no_grad():
                loss_val = criterion(model(X), y).item()
            Z[i, j] = loss_val

    # Restore original weights
    idx = 0
    for p in model.parameters():
        n_params = p.numel()
        p.data = theta_star[idx:idx + n_params].view(p.shape)
        idx += n_params

    return {
        "alpha_grid": A,
        "beta_grid": B,
        "loss_surface": Z,
        "model_info": model_info,
        "final_loss": final_loss,
    }
