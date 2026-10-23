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

    fig_json = fig.to_json()

    # Standalone, zero-dependency HTML template with resilient dynamic loader
    # Works in local browsers, GitHub Pages, and sandboxed proxy iframes like htmlpreview.github.io
    html_content = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>loss-topography — Interactive 3D Loss Landscape &amp; Optimizer Race</title>
    <style>
        * {{ box-sizing: border-box; }}
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            background-color: {BG_CANVAS};
            color: #F9FAFB;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            overflow: hidden;
        }}
        #top-bar {{
            background-color: {BG_PANEL};
            border-bottom: 1px solid {GRID_COLOR};
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 62px;
            position: relative;
            z-index: 10;
        }}
        #plot-viewport {{
            width: 100%;
            height: calc(100vh - 62px);
            position: relative;
            background-color: {BG_CANVAS};
        }}
        #loading-overlay {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            z-index: 50;
            background: rgba(17, 24, 39, 0.95);
            padding: 28px 40px;
            border-radius: 12px;
            border: 1px solid #374151;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            min-width: 280px;
        }}
        .spinner {{
            width: 44px;
            height: 44px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top-color: #3B82F6;
            animation: spin 0.9s cubic-bezier(0.4, 0, 0.2, 1) infinite;
            margin: 0 auto 16px auto;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        #plotly-div {{
            width: 100%;
            height: 100%;
        }}
    </style>
</head>
<body>
    <div id="top-bar">
        <div>
            <h2 style="margin: 0; color: #FFFFFF; font-size: 15px; font-weight: 600;">🏔️ loss-topography &bull; 3D Interactive Loss Surface Laboratory</h2>
            <p style="margin: 3px 0 0 0; color: {TEXT_LABEL}; font-size: 11px;">Click &amp; drag to rotate 360&deg; &bull; Scroll to zoom &bull; Double-click to reset &bull; Click legend to isolate optimizers</p>
        </div>
        <div>
            <a href="https://github.com/deepakj111/loss-topography" target="_blank" style="background: #2563EB; color: white; text-decoration: none; padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 500;">★ GitHub Repository</a>
        </div>
    </div>

    <div id="plot-viewport">
        <div id="loading-overlay">
            <div class="spinner"></div>
            <div style="font-size: 15px; font-weight: 600; color: #FFFFFF;">Initializing 3D Loss Topography...</div>
            <div id="loading-subtext" style="font-size: 12px; color: #60A5FA; margin-top: 6px;">Loading WebGL &amp; Plotly engine</div>
        </div>
        <div id="plotly-div"></div>
    </div>

    <script>
    (function() {{
        var figData = {fig_json};
        var div = document.getElementById('plotly-div');
        var loader = document.getElementById('loading-overlay');
        var subtext = document.getElementById('loading-subtext');

        function renderPlot() {{
            if (typeof Plotly === 'undefined') {{
                if (subtext) subtext.innerText = 'Waiting for Plotly runtime...';
                setTimeout(renderPlot, 100);
                return;
            }}
            if (subtext) subtext.innerText = 'Compiling WebGL shaders & rendering...';
            Plotly.newPlot(div, figData.data, figData.layout, {{responsive: true, displayModeBar: true}}).then(function() {{
                if (loader) loader.style.display = 'none';
            }}).catch(function(err) {{
                if (loader) {{
                    loader.innerHTML = '<div style="color:#EF4444;font-weight:600;font-size:16px;">WebGL Initialization Error</div>' +
                                       '<div style="color:#9CA3AF;font-size:12px;margin-top:8px;">' + (err.message || err) + '</div>' +
                                       '<div style="color:#6B7280;font-size:11px;margin-top:8px;">Ensure hardware acceleration is enabled in your browser settings.</div>';
                }}
            }});
        }}

        // Resilient script loader: injects Plotly dynamically if not already available in environment
        if (typeof Plotly !== 'undefined') {{
            renderPlot();
        }} else {{
            var s = document.createElement('script');
            s.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
            s.crossOrigin = 'anonymous';
            s.onload = renderPlot;
            s.onerror = function() {{
                if (subtext) subtext.innerText = 'Switching to fallback CDN...';
                var s2 = document.createElement('script');
                s2.src = 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js';
                s2.onload = renderPlot;
                s2.onerror = function() {{
                    if (loader) {{
                        loader.innerHTML = '<div style="color:#EF4444;font-weight:600;font-size:16px;">Failed to load Plotly CDN</div>' +
                                           '<div style="color:#9CA3AF;font-size:12px;margin-top:8px;">Please check your network connection or adblocker rules.</div>';
                    }}
                }};
                document.head.appendChild(s2);
            }};
            document.head.appendChild(s);
        }}
    }})();
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"    ✓ Saved {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")


if __name__ == "__main__":
    export_interactive_html()
