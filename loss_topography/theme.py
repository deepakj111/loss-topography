"""
Unified dark-mode styling, palettes, and Matplotlib / Plotly rcParams.

Design System: "Nerdy Cool" Dark-Mode Theme
============================================
Background canvas : #0B0F19  (deep obsidian navy)
Panel / subplot   : #111827  (slate dark)
Gridlines         : #1F2937  @ alpha 0.35
Text header       : #F9FAFB
Axis labels       : #9CA3AF
"""

from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Colour constants ───────────────────────────────────────────────
BG_CANVAS   = "#0B0F19"
BG_PANEL    = "#111827"
GRID_COLOR  = "#1F2937"
GRID_ALPHA  = 0.35
TEXT_HEADER = "#F9FAFB"
TEXT_LABEL  = "#9CA3AF"
TEXT_TICK   = "#6B7280"

# Optimizer marble palette (high-contrast neon)
OPTIMIZER_COLORS = {
    "SGD":            "#EF4444",   # Crimson Neon
    "Momentum":       "#F59E0B",   # Amber Gold
    "NAG":            "#10B981",   # Emerald Green
    "AdaGrad":        "#A855F7",   # Electric Purple
    "RMSProp":        "#EC4899",   # Hot Magenta
    "Adam":           "#06B6D4",   # Cyan Laser
    "AdamW":          "#3B82F6",   # Cobalt Glow
    "Damped Newton":  "#FFFFFF",   # Pure White Star
}

OPTIMIZER_ORDER = list(OPTIMIZER_COLORS.keys())


def _make_dark_cmap(name: str = "topo_dark"):
    """Custom cyan-to-magenta topographic elevation colormap."""
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.00, "#0B0F19"),
        (0.15, "#164E63"),
        (0.30, "#0891B2"),
        (0.45, "#06B6D4"),
        (0.55, "#22D3EE"),
        (0.65, "#A855F7"),
        (0.80, "#EC4899"),
        (0.90, "#F43F5E"),
        (1.00, "#FCD34D"),
    ]
    positions = [c[0] for c in colors]
    hex_colors = [c[1] for c in colors]
    rgb = [mpl.colors.to_rgb(h) for h in hex_colors]
    return LinearSegmentedColormap.from_list(name, list(zip(positions, rgb)))


DARK_CMAP = _make_dark_cmap()


def apply_mpl_theme():
    """Apply the dark-mode rcParams globally for Matplotlib."""
    plt.rcParams.update({
        "figure.facecolor":   BG_CANVAS,
        "axes.facecolor":     BG_PANEL,
        "savefig.facecolor":  BG_CANVAS,
        "axes.edgecolor":     GRID_COLOR,
        "axes.labelcolor":    TEXT_LABEL,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.grid":          True,
        "grid.color":         GRID_COLOR,
        "grid.alpha":         GRID_ALPHA,
        "grid.linewidth":     0.5,
        "xtick.color":        TEXT_TICK,
        "ytick.color":        TEXT_TICK,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "text.color":         TEXT_HEADER,
        "legend.facecolor":   BG_PANEL,
        "legend.edgecolor":   GRID_COLOR,
        "legend.fontsize":    9,
        "legend.framealpha":  0.85,
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Helvetica", "Arial"],
        "figure.dpi":         150,
        "savefig.dpi":        150,
        "animation.embed_limit": 100,
    })


def plotly_layout_defaults() -> dict:
    """Return a dict suitable for `fig.update_layout(**plotly_layout_defaults())`."""
    return dict(
        paper_bgcolor=BG_CANVAS,
        plot_bgcolor=BG_PANEL,
        font=dict(color=TEXT_HEADER, family="Inter, sans-serif", size=12),
        title_font=dict(size=16, color=TEXT_HEADER),
        legend=dict(bgcolor=BG_PANEL, bordercolor=GRID_COLOR, borderwidth=1,
                    font=dict(color=TEXT_LABEL, size=10)),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                   tickfont=dict(color=TEXT_TICK)),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                   tickfont=dict(color=TEXT_TICK)),
        margin=dict(l=50, r=30, t=60, b=50),
    )


def plotly_3d_scene(title: str = "") -> dict:
    """Return scene config for Plotly 3D surface plots."""
    axis_cfg = dict(
        backgroundcolor=BG_PANEL,
        gridcolor=GRID_COLOR,
        showbackground=True,
        tickfont=dict(color=TEXT_TICK, size=9),
    )
    return dict(
        scene=dict(
            xaxis=dict(**axis_cfg, title=dict(text="θ₁", font=dict(color=TEXT_LABEL, size=11))),
            yaxis=dict(**axis_cfg, title=dict(text="θ₂", font=dict(color=TEXT_LABEL, size=11))),
            zaxis=dict(**axis_cfg, title=dict(text="Loss", font=dict(color=TEXT_LABEL, size=11))),
            bgcolor=BG_CANVAS,
        ),
    )
