#!/usr/bin/env python3
"""
Exports a standalone, zero-dependency interactive 3D HTML visualization
into assets/interactive_3d_landscape.html.

Anyone can open this file in any web browser to rotate, pan, zoom, and inspect
the 3D loss surface and optimizer trajectories without installing Python or PyTorch.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from loss_topography.theme import (
    OPTIMIZER_COLORS, BG_CANVAS, BG_PANEL, GRID_COLOR,
    TEXT_HEADER, TEXT_LABEL, TEXT_TICK, plotly_layout_defaults,
)
from loss_topography.surfaces import MultiModal
from loss_topography.optimizers import SGD, MomentumSGD, RMSProp, Adam

ASSETS_DIR = os.path.join(ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def export_interactive_html():
    print("  🌐 Generating interactive_3d_landscape.html ...")

    surface = MultiModal()
    X, Y, Z = surface.meshgrid(n=80)
    Z_clip = np.clip(Z, np.percentile(Z, 2), np.percentile(Z, 95))

    # Run optimizers from high-elevation starting point
    x0, y0 = 2.8, 2.8
    n_steps = 150

    optimizers = {
        "SGD": SGD(lr=0.03),
        "Momentum": MomentumSGD(lr=0.015, beta=0.9),
        "RMSProp": RMSProp(lr=0.04),
        "Adam": Adam(lr=0.06),
    }

    logs = {name: opt.run(surface, x0, y0, n_steps=n_steps)
            for name, opt in optimizers.items()}

    # Create 3D Figure
    fig = go.Figure()

    # Add 3D surface
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z_clip,
            colorscale="Viridis",
            opacity=0.82,
            showscale=True,
            colorbar=dict(
                title=dict(text="Loss", font=dict(color=TEXT_LABEL, size=11)),
                tickfont=dict(color=TEXT_TICK, size=9),
                len=0.7,
                thickness=16,
                x=1.02,
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="#FFFFFF", project_z=True),
            ),
            hoverinfo="x+y+z",
            name="Loss Surface",
        )
    )

    # Add optimizer trajectories in 3D
    for name, log in logs.items():
        color = OPTIMIZER_COLORS[name]
        # Trajectory line
        fig.add_trace(
            go.Scatter3d(
                x=log.xs, y=log.ys, z=log.losses,
                mode="lines+markers",
                name=f"{name} (Final L={log.losses[-1]:.3f})",
                line=dict(color=color, width=5),
                marker=dict(
                    size=[3 if i < len(log.xs)-1 else 8 for i in range(len(log.xs))],
                    color=color,
                    symbol="circle",
                ),
                customdata=np.stack([
                    np.arange(len(log.xs)),
                    [log.path_efficiency] * len(log.xs),
                ], axis=-1),
                hovertemplate=(
                    f"<b>{name}</b><br>" +
                    "Step: %{customdata[0]}<br>" +
                    "θ₁: %{x:.3f}<br>" +
                    "θ₂: %{y:.3f}<br>" +
                    "Loss: %{z:.4f}<br>" +
                    "<extra></extra>"
                ),
            )
        )

    # Starting point marker
    z0 = float(surface.f_scalar(x0, y0))
    fig.add_trace(
        go.Scatter3d(
            x=[x0], y=[y0], z=[z0],
            mode="markers+text",
            name="Start Position (θ₀)",
            marker=dict(size=9, color="#EF4444", symbol="diamond"),
            text=["Start θ₀"],
            textposition="top center",
            textfont=dict(color="#EF4444", size=11),
            hoverinfo="text",
        )
    )

    # 3D Scene layout configuration
    axis_cfg = dict(
        backgroundcolor=BG_PANEL,
        gridcolor=GRID_COLOR,
        showbackground=True,
        tickfont=dict(color=TEXT_TICK, size=9),
    )
    scene_cfg = dict(
        scene=dict(
            xaxis=dict(**axis_cfg, title=dict(text="Parameter θ₁", font=dict(color=TEXT_LABEL, size=11))),
            yaxis=dict(**axis_cfg, title=dict(text="Parameter θ₂", font=dict(color=TEXT_LABEL, size=11))),
            zaxis=dict(**axis_cfg, title=dict(text="Loss Function L(θ)", font=dict(color=TEXT_LABEL, size=11))),
            bgcolor=BG_CANVAS,
            camera=dict(
                eye=dict(x=-1.5, y=-1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
    )

    layout = plotly_layout_defaults()
    layout["legend"].update(dict(
        x=0.02, y=0.88,
        bgcolor="rgba(18, 24, 38, 0.85)",
        bordercolor=GRID_COLOR,
        borderwidth=1,
        font=dict(color=TEXT_HEADER, size=11),
    ))
    layout["margin"] = dict(l=20, r=20, t=60, b=20)
    fig.update_layout(
        **layout,
        **scene_cfg,
        title=dict(
            text="<b>🏔️ loss-topography — Interactive 3D Loss Landscape & Optimizer Trajectories</b>",
            font=dict(size=18, color=TEXT_HEADER),
            x=0.03, y=0.97,
        ),
        height=780,
    )

    output_path = os.path.join(ASSETS_DIR, "interactive_3d_landscape.html")

    # Custom wrapper with modern dark UI styling and help banner
    html_content = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
    )

    # Inject stylish header bar and instructions
    styled_header = f"""
    <div style="background-color: {BG_PANEL}; border-bottom: 1px solid {GRID_COLOR}; padding: 12px 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin: 0; color: #FFFFFF; font-size: 16px; font-weight: 600;">🏔️ loss-topography &bull; 3D Interactive Loss Surface Laboratory</h2>
            <p style="margin: 4px 0 0 0; color: {TEXT_LABEL}; font-size: 12px;">Click & drag to rotate 360&deg; | Scroll to zoom | Double-click to reset view | Click legend items to toggle optimizers</p>
        </div>
        <div>
            <a href="https://github.com/deepakj111/loss-topography" target="_blank" style="background: #2563EB; color: white; text-decoration: none; padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 500;">★ GitHub Repository</a>
        </div>
    </div>
    """
    html_content = html_content.replace("<body>", f"<body style='margin:0; background-color:{BG_CANVAS};'>{styled_header}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"    ✓ Saved {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")


if __name__ == "__main__":
    export_interactive_html()
