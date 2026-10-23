#!/usr/bin/env python3
"""
Interactive 3D Visualizer & Physics Engine Generator for loss-topography.

Compiles a standalone, zero-dependency, rich WebGL 3D educational laboratory into
assets/interactive_3d_landscape.html.

Key Educational & Visual Improvements:
1. Uncluttered, High-Fidelity 3D Shading:
   - Smooth elevation colormap (Plasma / Viridis) with tuned ambient and specular lighting.
   - Contour lines projected cleanly onto the ground plane (no busy wireframes obstructing terrain).
   - Log-compressed elevation for high-dynamic-range surfaces (Rosenbrock, Beale) so deep
     ravines and valleys are immediately visible rather than flattened.
2. Prominent Geometric Landmarks:
   - 🎯 Global Minimum Beacon: Glowing gold target marker at exact analytical optimum θ*.
   - 📍 Starting Coordinates Beacon: Glowing red diamond at θ₀.
   - ⚠️ Saddle Point Landmark: Distinct warning marker showing saddle location & eigenvalues.
3. Zero-Lag 60 FPS Animation:
   - Separates static surface rendering from dynamic trajectory restyling.
   - Uses requestAnimationFrame with trace-specific restyle, avoiding heavy mesh rebuilds.
4. Curated Default Comparison:
   - Focuses by default on 3 pedagogical archetypes (SGD vs Momentum vs Adam),
     with instant toggle for NAG, AdaGrad, RMSProp, AdamW, Damped Newton.
5. Dual-View Mode:
   - 🌐 3D Perspective View (orbiting landscape)
   - 🗺️ 2D Contour Map (orthographic top-down view, classic Alec Radford / Distill style)
   - 📐 Side Elevation Profile (inspecting vertical slopes and barrier heights)
6. Synchronized Live Convergence Graph:
   - Real-time SVG chart tracking log-loss vs iteration for every active optimizer.
7. True Hydro Dynamic River Flow:
   - Animated stream of liquid particles cascading downhill with momentum and fading trails.
8. Educational "What This Teaches" Live Insight Guide:
   - Dynamic explanation box detailing the geometric challenges (condition number, curvature,
     ravine walls, saddle barriers) and why specific optimizers succeed or fail.
9. Click-to-Drop Starting Coordinates:
   - Click anywhere on the 3D surface to launch the optimizers from that exact point.
"""

from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ASSETS_DIR = os.path.join(ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def build_interactive_app_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>loss-topography — Interactive 3D Loss Surface & Optimizer Laboratory</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body {
            width: 100%;
            height: 100%;
            background-color: #0B0F19;
            color: #F9FAFB;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            overflow: hidden;
        }

        /* Top Bar */
        #header {
            height: 56px;
            background: #111827;
            border-bottom: 1px solid #1F2937;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 18px;
            z-index: 30;
            position: relative;
        }
        .header-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-title {
            font-size: 15px;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.2px;
        }
        .header-tag {
            background: rgba(37, 99, 235, 0.15);
            color: #60A5FA;
            border: 1px solid rgba(37, 99, 235, 0.35);
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
        }
        .toggle-group {
            display: flex;
            background: #1F2937;
            border-radius: 8px;
            padding: 3px;
            gap: 4px;
        }
        .t-btn {
            background: transparent;
            border: none;
            color: #9CA3AF;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .t-btn.active {
            background: #2563EB;
            color: #FFFFFF;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        .btn-link {
            background: #1F2937;
            color: #E5E7EB;
            text-decoration: none;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid #374151;
            transition: all 0.15s;
        }
        .btn-link:hover { background: #374151; color: #FFF; }

        /* Main Workspace */
        #workspace {
            display: flex;
            height: calc(100vh - 56px);
            width: 100%;
            position: relative;
        }

        /* Sidebar */
        #sidebar {
            width: 340px;
            background: rgba(17, 24, 39, 0.95);
            border-right: 1px solid #1F2937;
            overflow-y: auto;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            z-index: 20;
            backdrop-filter: blur(12px);
        }
        .card {
            background: #161F30;
            border: 1px solid #1F2937;
            border-radius: 10px;
            padding: 12px 14px;
        }
        .card-header {
            font-size: 11px;
            font-weight: 700;
            color: #9CA3AF;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .form-row {
            margin-bottom: 8px;
        }
        .form-row:last-child { margin-bottom: 0; }
        .label-row {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #D1D5DB;
            margin-bottom: 3px;
        }
        .label-row span.val {
            color: #60A5FA;
            font-family: monospace;
            font-weight: 600;
        }
        select {
            width: 100%;
            background: #0B0F19;
            border: 1px solid #374151;
            color: #F9FAFB;
            padding: 7px 10px;
            border-radius: 6px;
            font-size: 12px;
            outline: none;
            cursor: pointer;
        }
        select:focus { border-color: #3B82F6; }
        input[type="range"] {
            width: 100%;
            accent-color: #3B82F6;
            height: 5px;
            background: #374151;
            border-radius: 3px;
            outline: none;
        }

        /* Optimizer Toggle Chips */
        .chip-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
        }
        .opt-chip {
            background: #0B0F19;
            border: 1px solid #374151;
            padding: 6px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            color: #6B7280;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            user-select: none;
            transition: all 0.15s;
        }
        .opt-chip.active {
            border-color: currentColor;
            color: #FFFFFF;
            background: rgba(255, 255, 255, 0.05);
        }
        .chip-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }

        /* Playback Buttons */
        .play-row {
            display: flex;
            gap: 6px;
            margin-top: 8px;
        }
        .btn-play {
            flex: 1.2;
            background: #2563EB;
            color: white;
            border: none;
            padding: 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: background 0.15s;
        }
        .btn-play:hover { background: #1D4ED8; }
        .btn-sub {
            background: #1F2937;
            color: #D1D5DB;
            border: 1px solid #374151;
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }
        .btn-sub:hover { background: #374151; color: white; }

        /* Pedagogical Insight Box */
        #insight-box {
            background: rgba(30, 41, 59, 0.75);
            border-left: 3px solid #3B82F6;
            padding: 10px 12px;
            border-radius: 6px;
            font-size: 11px;
            line-height: 1.45;
            color: #CBD5E1;
        }
        #insight-box b { color: #93C5FD; }
        #insight-box code { background: rgba(0,0,0,0.3); padding: 1px 4px; border-radius: 3px; font-family: monospace; }

        /* Preset buttons */
        .preset-grid {
            display: flex;
            gap: 4px;
            margin-top: 6px;
            flex-wrap: wrap;
        }
        .btn-preset {
            background: #1F2937;
            color: #9CA3AF;
            border: 1px solid #374151;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }
        .btn-preset:hover { background: #374151; color: #FFF; }

        /* Viewport Canvas */
        #viewport {
            flex: 1;
            height: 100%;
            position: relative;
            background: #0B0F19;
        }
        #plotly-canvas {
            width: 100%;
            height: 100%;
        }

        /* Top View Toolbar */
        .viewport-toolbar {
            position: absolute;
            top: 14px;
            left: 18px;
            z-index: 15;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .view-pill-group {
            background: rgba(17, 24, 39, 0.88);
            border: 1px solid #1F2937;
            border-radius: 8px;
            padding: 3px;
            display: flex;
            gap: 4px;
            backdrop-filter: blur(8px);
        }
        .view-pill {
            background: transparent;
            border: none;
            color: #9CA3AF;
            padding: 5px 11px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }
        .view-pill.active {
            background: #374151;
            color: #FFFFFF;
        }
        .hint-badge {
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid #1F2937;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 11px;
            color: #9CA3AF;
            pointer-events: none;
            backdrop-filter: blur(8px);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Telemetry HUD */
        #telemetry-hud {
            position: absolute;
            bottom: 16px;
            right: 18px;
            width: 480px;
            max-width: calc(100vw - 380px);
            background: rgba(17, 24, 39, 0.94);
            border: 1px solid #1F2937;
            border-radius: 12px;
            padding: 12px 14px;
            backdrop-filter: blur(12px);
            z-index: 25;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
        }
        .hud-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid #1F2937;
        }
        .hud-title {
            font-size: 11px;
            font-weight: 700;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .hud-meta {
            display: flex;
            gap: 10px;
            font-size: 11px;
            color: #9CA3AF;
            font-family: monospace;
        }
        .hud-meta span.hi { color: #34D399; font-weight: 700; }
        .hud-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }
        .hud-table th {
            text-align: left;
            color: #9CA3AF;
            font-weight: 600;
            padding: 2px 4px;
            border-bottom: 1px solid #1F2937;
        }
        .hud-table td {
            padding: 3px 4px;
            color: #E5E7EB;
            font-family: monospace;
        }

        /* Convergence Mini Chart */
        #convergence-chart-wrap {
            margin-top: 8px;
            padding-top: 6px;
            border-top: 1px solid #1F2937;
        }
        .chart-label-row {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #9CA3AF;
            margin-bottom: 4px;
        }
        #convergence-svg {
            width: 100%;
            height: 60px;
            background: #0B0F19;
            border-radius: 6px;
            border: 1px solid #1F2937;
        }

        /* Loading Screen */
        #loading-screen {
            position: absolute;
            inset: 0;
            background: #0B0F19;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 100;
        }
        .spinner {
            width: 44px;
            height: 44px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top-color: #3B82F6;
            animation: spin 0.8s ease-in-out infinite;
            margin-bottom: 14px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <!-- Top Header -->
    <header id="header">
        <div class="header-left">
            <span style="font-size: 20px;">🏔️</span>
            <div>
                <span class="header-title">loss-topography</span>
                <span class="header-tag">Interactive ML Optimization Laboratory</span>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div class="toggle-group">
                <button id="mode-race" class="t-btn active" onclick="setMode('race')">🏁 Optimizer Race</button>
                <button id="mode-river" class="t-btn" onclick="setMode('river')">🌊 Hydro River Flow</button>
            </div>
            <a href="https://github.com/deepakj111/loss-topography" target="_blank" class="btn-link">★ GitHub</a>
        </div>
    </header>

    <!-- App Body -->
    <div id="workspace">
        <!-- Control Sidebar -->
        <aside id="sidebar">
            <!-- Landscape Selector -->
            <div class="card">
                <div class="card-header">
                    <span>Loss Landscape</span>
                </div>
                <div class="form-row">
                    <select id="surface-select" onchange="onSurfaceChange()">
                        <option value="quadratic" selected>🥣 Ill-Conditioned Quadratic Bowl (Distill.pub Benchmark, κ = 20)</option>
                        <option value="rosenbrock">🍌 Rosenbrock Banana Valley (Anisotropic Ravine, κ ≫ 10³)</option>
                        <option value="regsaddle">🕳️ Regularized Saddle (Zero Gradient &amp; Negative Curvature)</option>
                        <option value="beale">🏔️ Beale Function (Flat Wings &amp; Corner Ravine)</option>
                        <option value="multimodal">🌊 Multi-Modal Potential (Competing Basins &amp; Mountain Pass)</option>
                        <option value="rastrigin">🌀 Rastrigin Surface (Dense Minima Lattice)</option>
                    </select>
                </div>
            </div>

            <!-- Pedagogical Insight Box -->
            <div id="insight-box">
                <!-- Dynamically populated by JS -->
            </div>

            <!-- Playback Controls -->
            <div class="card">
                <div class="card-header">
                    <span>Simulation Playback</span>
                    <span id="step-counter" style="color: #60A5FA; font-family: monospace;">Step: 0 / 120</span>
                </div>
                <div class="form-row">
                    <input type="range" id="scrubber" min="0" max="120" value="0" oninput="onScrub(this.value)">
                </div>
                <div class="play-row">
                    <button id="btn-play" class="btn-play" onclick="togglePlay()">
                        <span id="play-icon">▶</span> <span id="play-text">Play Race</span>
                    </button>
                    <button class="btn-sub" onclick="stepForward()">⏭ +5 Steps</button>
                    <button class="btn-sub" onclick="resetSim()">⏮ Reset</button>
                </div>
                <div class="form-row" style="margin-top: 8px; display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 11px; color: #9CA3AF;">Speed:</span>
                    <div style="display: flex; gap: 3px;">
                        <button class="btn-sub" style="padding: 2px 7px; font-size: 10px;" id="spd-0.5" onclick="setSpeed(0.5)">0.5x</button>
                        <button class="btn-sub" style="padding: 2px 7px; font-size: 10px; background: #2563EB; color:white;" id="spd-1" onclick="setSpeed(1)">1x</button>
                        <button class="btn-sub" style="padding: 2px 7px; font-size: 10px;" id="spd-2" onclick="setSpeed(2)">2x</button>
                        <button class="btn-sub" style="padding: 2px 7px; font-size: 10px;" id="spd-4" onclick="setSpeed(4)">4x</button>
                    </div>
                </div>
            </div>

            <!-- Optimizers Toggle (Race Mode) -->
            <div class="card" id="opt-selection-group">
                <div class="card-header">
                    <span>Active Competitors</span>
                    <button class="btn-sub" style="padding: 2px 6px; font-size: 9px;" onclick="toggleAllOptimizers()">Toggle All</button>
                </div>
                <div class="chip-grid" id="opt-chips-container">
                    <!-- Populated by JS -->
                </div>
            </div>

            <!-- Hyperparameters -->
            <div class="card">
                <div class="card-header">
                    <span>Hyperparameters</span>
                </div>
                <div class="form-row">
                    <div class="label-row">
                        <span>Learning Rate (η)</span>
                        <span class="val" id="lr-val">0.050</span>
                    </div>
                    <input type="range" id="lr-slider" min="0.001" max="0.15" step="0.001" value="0.05" oninput="updateHyperparams()">
                </div>
                <div class="form-row">
                    <div class="label-row">
                        <span>Momentum (β)</span>
                        <span class="val" id="beta-val">0.85</span>
                    </div>
                    <input type="range" id="beta-slider" min="0.0" max="0.98" step="0.01" value="0.85" oninput="updateHyperparams()">
                </div>
                <div class="form-row">
                    <div class="label-row">
                        <span>Total Steps (N)</span>
                        <span class="val" id="steps-val">120</span>
                    </div>
                    <input type="range" id="steps-slider" min="30" max="250" step="10" value="120" oninput="updateHyperparams()">
                </div>
            </div>

            <!-- Starting Point Coordinates -->
            <div class="card">
                <div class="card-header">
                    <span>Start Position (θ₀)</span>
                    <button class="btn-sub" style="padding: 2px 6px; font-size: 9px;" onclick="randomizeStart()">🎲 Randomize</button>
                </div>
                <div class="form-row">
                    <div class="label-row">
                        <span>Coordinate θ₁ (x₀)</span>
                        <span class="val" id="x0-val">3.50</span>
                    </div>
                    <input type="range" id="x0-slider" min="-4" max="4" step="0.05" value="3.5" oninput="updateCoordinates()">
                </div>
                <div class="form-row">
                    <div class="label-row">
                        <span>Coordinate θ₂ (y₀)</span>
                        <span class="val" id="y0-val">1.50</span>
                    </div>
                    <input type="range" id="y0-slider" min="-3" max="3" step="0.05" value="1.5" oninput="updateCoordinates()">
                </div>
                <div id="presets-container" class="preset-grid">
                    <!-- Populated dynamically per surface -->
                </div>
            </div>
        </aside>

        <!-- Main Viewport -->
        <main id="viewport">
            <div class="viewport-toolbar">
                <div class="view-pill-group">
                    <button id="view-3d" class="view-pill active" onclick="setViewMode('3d')">🌐 3D Perspective</button>
                    <button id="view-2d" class="view-pill" onclick="setViewMode('2d')">🗺️ 2D Contour Map</button>
                    <button id="view-side" class="view-pill" onclick="setViewMode('side')">📐 Side Elevation</button>
                </div>
                <div class="hint-badge">
                    <span>🖱️ Click anywhere on terrain to drop θ₀ &bull; Left-drag to orbit &bull; Scroll to zoom</span>
                </div>
            </div>

            <div id="plotly-canvas"></div>

            <!-- Floating HUD Telemetry & Synchronized Chart -->
            <div id="telemetry-hud">
                <div class="hud-head">
                    <div class="hud-title">
                        <span>📊 Live Optimization Telemetry</span>
                    </div>
                    <div class="hud-meta">
                        <span>Step Latency: <span class="hi" id="hud-latency">~2.4 µs</span></span>
                        <span>Global Min: <span class="hi" id="hud-fstar">0.00</span></span>
                    </div>
                </div>
                <table class="hud-table">
                    <thead>
                        <tr>
                            <th>Optimizer</th>
                            <th>Loss L(θ)</th>
                            <th>Dist to Min</th>
                            <th>Path Eff.</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="telemetry-tbody">
                        <!-- Populated dynamically -->
                    </tbody>
                </table>

                <!-- Convergence Mini Chart -->
                <div id="convergence-chart-wrap">
                    <div class="chart-label-row">
                        <span>📉 Convergence Rate: log₁₀(Loss) vs Step</span>
                        <span id="chart-step-lbl" style="font-family: monospace;">Step 0</span>
                    </div>
                    <svg id="convergence-svg" viewBox="0 0 450 60" preserveAspectRatio="none">
                        <!-- Rendered dynamically -->
                    </svg>
                </div>
            </div>
        </main>
    </div>

    <!-- Initial Loading Screen -->
    <div id="loading-screen">
        <div class="spinner"></div>
        <div style="font-size: 15px; font-weight: 600; color: #FFFFFF;">Initializing 3D WebGL Laboratory...</div>
        <div id="loading-status" style="font-size: 12px; color: #60A5FA; margin-top: 6px;">Loading Plotly 3D engine</div>
    </div>

    <script>
    // ═══════════════════════════════════════════════════════════════════
    // Mathematical Landscapes, Gradients, and Elevation Transforms
    // ═══════════════════════════════════════════════════════════════════
    const SURFACES = {
        quadratic: {
            name: "Ill-Conditioned Quadratic Bowl",
            fStar: 0.0,
            fStarX: 0.0,
            fStarY: 0.0,
            saddlePoint: null,
            xRange: [-4.0, 4.0],
            yRange: [-2.0, 2.0],
            defaultX0: 3.50,
            defaultY0: 1.50,
            defaultLr: 0.050,
            defaultBeta: 0.85,
            presets: [
                { label: "Steep Wall (y=1.8)", x: 3.5, y: 1.8 },
                { label: "Valley Floor (y=0.2)", x: 3.5, y: 0.2 },
                { label: "Symmetric Corner", x: -3.0, y: -1.5 }
            ],
            insight: "<b>Pedagogical Insight (Distill.pub Benchmark):</b> Canonical anisotropic bowl <code>f(x,y) = 0.5(x² + 20y²)</code> with condition number <b>κ = 20</b>. Along y (steep wall), curvature is 20, causing <b>SGD</b> to oscillate violently back and forth like a ping-pong ball while crawling along the flat x-axis. <b>Momentum</b> cancels out the alternating y-oscillations and builds velocity down the valley floor. <b>Adam</b> scales coordinates to cut diagonally straight to (0, 0).",
            f: function(x, y) {
                return 0.5 * (x * x + 20.0 * y * y);
            },
            grad: function(x, y) {
                return [x, 20.0 * y];
            },
            renderZ: function(val) {
                return Math.min(25.0, val * 0.5);
            }
        },
        rosenbrock: {
            name: "Rosenbrock Banana Valley",
            fStar: 0.0,
            fStarX: 1.0,
            fStarY: 1.0,
            saddlePoint: null,
            xRange: [-2.0, 2.0],
            yRange: [-1.0, 3.0],
            defaultX0: -1.20,
            defaultY0: 1.50,
            defaultLr: 0.0025,
            defaultBeta: 0.88,
            presets: [
                { label: "Steep Flank (-1.2, 1.5)", x: -1.2, y: 1.5 },
                { label: "Ravine Bend (0.0, 0.2)", x: 0.0, y: 0.2 },
                { label: "Near Optimum (0.8, 0.6)", x: 0.8, y: 0.6 }
            ],
            insight: "<b>Pedagogical Insight:</b> Classic ill-conditioned parabolic canyon (condition number κ ≫ 10³). The curved valley floor follows <code>y = x²</code> with the global minimum at (1, 1). First-order <b>SGD</b> bounces violently between the steep canyon walls, making almost zero forward progress along the curved riverbed. <b>Adam</b> rescales coordinate variance, and <b>Momentum</b> navigates the parabolic turn.",
            f: function(x, y) {
                var a = 1.0, b = 100.0;
                return (a - x)*(a - x) + b * Math.pow(y - x*x, 2);
            },
            grad: function(x, y) {
                var a = 1.0, b = 100.0;
                var gx = -2.0*(a - x) - 4.0*b*x*(y - x*x);
                var gy = 2.0*b*(y - x*x);
                return [gx, gy];
            },
            renderZ: function(val) {
                return Math.log10(1.0 + Math.max(0, val)) * 3.0;
            }
        },
        regsaddle: {
            name: "Regularized Saddle",
            fStar: -2.50,
            fStarX: 0.0,
            fStarY: 2.236, // (0, +√5)
            saddlePoint: { x: 0.0, y: 0.0, label: "⚠️ Saddle Point (0, 0)" },
            xRange: [-3.0, 3.0],
            yRange: [-3.0, 3.0],
            defaultX0: 0.05,
            defaultY0: 0.10,
            defaultLr: 0.035,
            defaultBeta: 0.85,
            presets: [
                { label: "Saddle Crest (0.05, 0.10)", x: 0.05, y: 0.10 },
                { label: "Plateau Ridge (0.02, 0.02)", x: 0.02, y: 0.02 },
                { label: "High Flank (2.0, 2.0)", x: 2.0, y: 2.0 }
            ],
            insight: "<b>Pedagogical Insight (Alec Radford Landmark):</b> Strict saddle point at (0, 0) with ∇f = 0 and Hessian eigenvalues λ₁ = +2, λ₂ = -2. <b>SGD</b> crawls to a complete standstill on the plateau because gradients vanish. <b>Momentum</b> and <b>NAG</b> build exponential kinetic velocity along the negative curvature direction (y-axis) to escape the saddle, plunging into the twin minima at (0, ±√5) with f* = -2.50.",
            f: function(x, y) {
                return x*x - y*y + 0.1*Math.pow(x, 4) + 0.1*Math.pow(y, 4);
            },
            grad: function(x, y) {
                return [2*x + 0.4*Math.pow(x, 3), -2*y + 0.4*Math.pow(y, 3)];
            },
            renderZ: function(val) { return val; }
        },
        beale: {
            name: "Beale Function",
            fStar: 0.0,
            fStarX: 3.0,
            fStarY: 0.5,
            saddlePoint: null,
            xRange: [-2.0, 4.0],
            yRange: [-2.0, 2.0],
            defaultX0: 2.00,
            defaultY0: 2.00,
            defaultLr: 0.012,
            defaultBeta: 0.85,
            presets: [
                { label: "Corner Wing (2.0, 2.0)", x: 2.0, y: 2.0 },
                { label: "Flat Plateau (-1.0, -1.0)", x: -1.0, y: -1.0 },
                { label: "Near Basin (2.8, 0.8)", x: 2.8, y: 0.8 }
            ],
            insight: "<b>Pedagogical Insight:</b> Classic test surface with expansive flat outer plateaus and a sharp corner ravine leading to (3, 0.5). High polynomial gradients (O(x²y³)) cause constant-rate SGD to either stall on the flat wings or blow up in the corner, whereas adaptive methods (<b>AdaGrad</b>, <b>RMSProp</b>, <b>Adam</b>) automatically dampen explosive steps.",
            f: function(x, y) {
                var t1 = 1.5 - x + x*y;
                var t2 = 2.25 - x + x*y*y;
                var t3 = 2.625 - x + x*y*y*y;
                return t1*t1 + t2*t2 + t3*t3;
            },
            grad: function(x, y) {
                var t1 = 1.5 - x + x*y;
                var t2 = 2.25 - x + x*y*y;
                var t3 = 2.625 - x + x*y*y*y;
                return [2*t1*(y - 1) + 2*t2*(y*y - 1) + 2*t3*(y*y*y - 1),
                        2*t1*x + 4*t2*x*y + 6*t3*x*y*y];
            },
            renderZ: function(val) {
                return Math.log10(1.0 + Math.max(0, val)) * 2.8;
            }
        },
        multimodal: {
            name: "Multi-Modal Potential",
            fStar: -6.55,
            fStarX: 1.35,
            fStarY: -1.35,
            saddlePoint: null,
            xRange: [-4.0, 4.0],
            yRange: [-4.0, 4.0],
            defaultX0: 2.50,
            defaultY0: 2.50,
            defaultLr: 0.030,
            defaultBeta: 0.85,
            presets: [
                { label: "High Ridge (2.5, 2.5)", x: 2.5, y: 2.5 },
                { label: "Local Trap (-1.2, 1.2)", x: -1.2, y: 1.2 },
                { label: "Mountain Pass (0.0, 0.0)", x: 0.0, y: 0.0 }
            ],
            insight: "<b>Pedagogical Insight:</b> Competing basins separated by a high mountain ridge. Demonstrates how <b>Momentum</b> carries sufficient kinetic inertia to cross elevated passes into the global basin (f* ≈ -6.55), while standard first-order <b>SGD</b> gets trapped in the nearest sub-optimal local minimum (f ≈ -3.8).",
            f: function(x, y) {
                var r2 = x*x + y*y;
                var z = -(Math.cos(x)*Math.sin(y)*Math.exp(-r2/8.0) + 0.4*Math.cos(1.8*x) + 0.4*Math.sin(1.8*y) + 0.08*r2);
                return z * 3.5;
            },
            grad: function(x, y) {
                var eps = 1e-5;
                return [(this.f(x + eps, y) - this.f(x - eps, y)) / (2*eps),
                        (this.f(x, y + eps) - this.f(x, y - eps)) / (2*eps)];
            },
            renderZ: function(val) { return val; }
        },
        rastrigin: {
            name: "Rastrigin Surface",
            fStar: 0.0,
            fStarX: 0.0,
            fStarY: 0.0,
            saddlePoint: null,
            xRange: [-4.0, 4.0],
            yRange: [-4.0, 4.0],
            defaultX0: 2.50,
            defaultY0: 2.50,
            defaultLr: 0.020,
            defaultBeta: 0.88,
            presets: [
                { label: "Outer Rim (2.5, 2.5)", x: 2.5, y: 2.5 },
                { label: "Mid Lattice (1.5, -1.5)", x: 1.5, y: -1.5 }
            ],
            insight: "<b>Pedagogical Insight:</b> Highly non-convex field riddled with dense lattices of cosine ripples. First-order gradient descent gets trapped in the nearest local minimum in fewer than 15 steps. <b>Momentum</b> provides the kinetic inertia required to skip over ripple barriers toward the central global bowl.",
            f: function(x, y) {
                return 20.0 + (x*x - 10*Math.cos(2*Math.PI*x)) + (y*y - 10*Math.cos(2*Math.PI*y));
            },
            grad: function(x, y) {
                return [2*x + 20*Math.PI*Math.sin(2*Math.PI*x),
                        2*y + 20*Math.PI*Math.sin(2*Math.PI*y)];
            },
            renderZ: function(val) { return val * 0.15; }
        }
    };

    // ═══════════════════════════════════════════════════════════════════
    // Optimizer Configuration (Default: 3 Archetypes to prevent clutter)
    // ═══════════════════════════════════════════════════════════════════
    const OPT_CONFIG = {
        "SGD":           { color: "#94A3B8", active: true },
        "Momentum":      { color: "#3B82F6", active: true },
        "Adam":          { color: "#F59E0B", active: true },
        "NAG":           { color: "#06B6D4", active: false },
        "AdaGrad":       { color: "#8B5CF6", active: false },
        "RMSProp":       { color: "#10B981", active: false },
        "AdamW":         { color: "#EC4899", active: false },
        "Damped Newton": { color: "#EF4444", active: false }
    };

    // ═══════════════════════════════════════════════════════════════════
    // State Management
    // ═══════════════════════════════════════════════════════════════════
    var state = {
        mode: "race",
        viewMode: "3d",
        surfaceKey: "quadratic",
        x0: 3.50,
        y0: 1.50,
        lr: 0.050,
        beta: 0.85,
        nSteps: 120,
        currentStep: 0,
        isPlaying: false,
        speed: 1.0,
        timer: null,
        simData: {},
        riverData: [],
        lastFrameTime: 0
    };

    // ═══════════════════════════════════════════════════════════════════
    // Mathematical Optimization Engine
    // ═══════════════════════════════════════════════════════════════════
    function computeHessian(surf, x, y) {
        var eps = 1e-4;
        var f0 = surf.f(x, y);
        var fxx = (surf.f(x + eps, y) - 2 * f0 + surf.f(x - eps, y)) / (eps * eps);
        var fyy = (surf.f(x, y + eps) - 2 * f0 + surf.f(x, y - eps)) / (eps * eps);
        var fxy = (surf.f(x + eps, y + eps) - surf.f(x + eps, y - eps) - surf.f(x - eps, y + eps) + surf.f(x - eps, y - eps)) / (4 * eps * eps);
        return [fxx, fxy, fxy, fyy];
    }

    function runOptimizer(name, surf, x0, y0, nSteps, lr, beta) {
        var xs = [x0], ys = [y0], losses = [surf.f(x0, y0)], renderZs = [surf.renderZ(surf.f(x0, y0))];
        var x = x0, y = y0;
        var vx = 0, vy = 0, mx = 0, my = 0, Gx = 0, Gy = 0;
        var epsStep = null;
        var totalDist = 0;
        var status = "Searching";

        var t0 = performance.now();

        for (var t = 0; t < nSteps; t++) {
            var g = [0, 0];
            if (name === "NAG") {
                var xAhead = x - beta * lr * vx;
                var yAhead = y - beta * lr * vy;
                g = surf.grad(xAhead, yAhead);
            } else {
                g = surf.grad(x, y);
            }

            var gNorm = Math.hypot(g[0], g[1]);
            if (gNorm > 120.0) {
                g[0] *= 120.0 / gNorm;
                g[1] *= 120.0 / gNorm;
            }

            var dx = 0, dy = 0;

            if (name === "SGD") {
                dx = -lr * g[0];
                dy = -lr * g[1];
            } else if (name === "Momentum" || name === "NAG") {
                vx = beta * vx + g[0];
                vy = beta * vy + g[1];
                dx = -lr * vx;
                dy = -lr * vy;
            } else if (name === "AdaGrad") {
                Gx += g[0] * g[0];
                Gy += g[1] * g[1];
                dx = - (lr * 3.5 / (Math.sqrt(Gx) + 1e-7)) * g[0];
                dy = - (lr * 3.5 / (Math.sqrt(Gy) + 1e-7)) * g[1];
            } else if (name === "RMSProp") {
                vx = 0.9 * vx + 0.1 * g[0] * g[0];
                vy = 0.9 * vy + 0.1 * g[1] * g[1];
                dx = - (lr / (Math.sqrt(vx) + 1e-7)) * g[0];
                dy = - (lr / (Math.sqrt(vy) + 1e-7)) * g[1];
            } else if (name === "Adam" || name === "AdamW") {
                mx = 0.9 * mx + 0.1 * g[0];
                my = 0.9 * my + 0.1 * g[1];
                vx = 0.999 * vx + 0.001 * g[0] * g[0];
                vy = 0.999 * vy + 0.001 * g[1] * g[1];
                var mHatX = mx / (1.0 - Math.pow(0.9, t + 1));
                var mHatY = my / (1.0 - Math.pow(0.9, t + 1));
                var vHatX = vx / (1.0 - Math.pow(0.999, t + 1));
                var vHatY = vy / (1.0 - Math.pow(0.999, t + 1));
                var decay = (name === "AdamW") ? lr * 0.01 : 0.0;
                dx = - decay * x - (lr * 1.5 / (Math.sqrt(vHatX) + 1e-7)) * mHatX;
                dy = - decay * y - (lr * 1.5 / (Math.sqrt(vHatY) + 1e-7)) * mHatY;
            } else if (name === "Damped Newton") {
                var H = computeHessian(surf, x, y);
                var mu = 0.8;
                var h11 = H[0] + mu, h12 = H[1], h22 = H[3] + mu;
                var det = h11 * h22 - h12 * h12;
                if (Math.abs(det) > 1e-6) {
                    var inv11 = h22 / det, inv12 = -h12 / det, inv22 = h11 / det;
                    dx = - (inv11 * g[0] + inv12 * g[1]) * 0.6;
                    dy = - (inv12 * g[0] + inv22 * g[1]) * 0.6;
                } else {
                    dx = -lr * g[0];
                    dy = -lr * g[1];
                }
            }

            var stepDist = Math.hypot(dx, dy);
            totalDist += stepDist;
            x += dx;
            y += dy;

            var l = surf.f(x, y);
            xs.push(x);
            ys.push(y);
            losses.push(l);
            renderZs.push(surf.renderZ(l));

            if (epsStep === null && Math.abs(l - surf.fStar) <= 0.02) {
                epsStep = t + 1;
            }
        }

        var netDist = Math.hypot(x - x0, y - y0);
        var pathEff = totalDist > 1e-7 ? Math.min(1.0, netDist / totalDist) : 1.0;
        var elapsedMs = performance.now() - t0;

        return {
            xs: xs,
            ys: ys,
            losses: losses,
            renderZs: renderZs,
            epsStep: epsStep,
            pathEff: pathEff,
            elapsedMs: elapsedMs
        };
    }

    function runRiverSimulation(surf, nStreams, nSteps, lr) {
        var streams = [];
        var bounds = surf.xRange;
        var spanX = bounds[1] - bounds[0];
        var spanY = surf.yRange[1] - surf.yRange[0];

        for (var i = 0; i < nStreams; i++) {
            var frac = (i + 0.5) / nStreams;
            var rx0 = bounds[0] + frac * spanX;
            var ry0 = surf.yRange[1] - 0.12 * spanY * Math.sin(frac * Math.PI);

            var sxs = [rx0], sys = [ry0], szs = [surf.renderZ(surf.f(rx0, ry0))], slosses = [surf.f(rx0, ry0)];
            var cx = rx0, cy = ry0;
            var vx = 0, vy = 0;

            for (var t = 0; t < nSteps; t++) {
                var g = surf.grad(cx, cy);
                var gNorm = Math.hypot(g[0], g[1]);
                if (gNorm > 80.0) { g[0] *= 80.0 / gNorm; g[1] *= 80.0 / gNorm; }

                vx = 0.92 * vx + 0.08 * g[0] + (Math.random() - 0.5) * 0.08;
                vy = 0.92 * vy + 0.08 * g[1] + (Math.random() - 0.5) * 0.08;

                cx -= lr * 1.5 * vx;
                cy -= lr * 1.5 * vy;

                var l = surf.f(cx, cy);
                sxs.push(cx);
                sys.push(cy);
                szs.push(surf.renderZ(l));
                slosses.push(l);
            }
            streams.push({ xs: sxs, ys: sys, renderZs: szs, losses: slosses });
        }
        return streams;
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3D Mesh Topography Generation (Optimized 40x40, zero wireframe)
    // ═══════════════════════════════════════════════════════════════════
    function generateMesh(surf, resolution) {
        var xR = surf.xRange, yR = surf.yRange;
        var xs = [], ys = [], zs = [];
        var dx = (xR[1] - xR[0]) / (resolution - 1);
        var dy = (yR[1] - yR[0]) / (resolution - 1);

        for (var i = 0; i < resolution; i++) {
            xs.push(xR[0] + i * dx);
            ys.push(yR[0] + i * dy);
        }

        for (var j = 0; j < resolution; j++) {
            var row = [];
            for (var i = 0; i < resolution; i++) {
                row.push(surf.renderZ(surf.f(xs[i], ys[j])));
            }
            zs.push(row);
        }
        return { X: xs, Y: ys, Z: zs };
    }

    // ═══════════════════════════════════════════════════════════════════
    // UI Setup & Event Handlers
    // ═══════════════════════════════════════════════════════════════════
    function buildOptimizerChips() {
        var container = document.getElementById('opt-chips-container');
        container.innerHTML = "";
        for (var name in OPT_CONFIG) {
            var cfg = OPT_CONFIG[name];
            var chip = document.createElement('div');
            chip.className = "opt-chip" + (cfg.active ? " active" : "");
            chip.id = "chip-" + name.replace(/\\s+/g, '-');
            chip.style.color = cfg.color;
            chip.onclick = (function(n) {
                return function() { toggleOpt(n); };
            })(name);

            chip.innerHTML = '<span class="chip-dot"></span><span>' + name + '</span>';
            container.appendChild(chip);
        }
    }

    function toggleOpt(name) {
        OPT_CONFIG[name].active = !OPT_CONFIG[name].active;
        var chip = document.getElementById("chip-" + name.replace(/\\s+/g, '-'));
        if (OPT_CONFIG[name].active) chip.classList.add('active');
        else chip.classList.remove('active');
        recomputeSim(true);
    }

    function toggleAllOptimizers() {
        var anyActive = Object.values(OPT_CONFIG).some(function(c) { return c.active; });
        for (var name in OPT_CONFIG) {
            OPT_CONFIG[name].active = !anyActive;
            var chip = document.getElementById("chip-" + name.replace(/\\s+/g, '-'));
            if (chip) {
                if (!anyActive) chip.classList.add('active');
                else chip.classList.remove('active');
            }
        }
        recomputeSim(true);
    }

    function setMode(mode) {
        state.mode = mode;
        document.getElementById('mode-race').classList.toggle('active', mode === 'race');
        document.getElementById('mode-river').classList.toggle('active', mode === 'river');
        document.getElementById('opt-selection-group').style.display = (mode === 'race' ? 'block' : 'none');
        document.getElementById('play-text').innerText = (mode === 'race' ? 'Play Race' : 'Flow River');
        recomputeSim(true);
    }

    function setViewMode(view) {
        state.viewMode = view;
        document.getElementById('view-3d').classList.toggle('active', view === '3d');
        document.getElementById('view-2d').classList.toggle('active', view === '2d');
        document.getElementById('view-side').classList.toggle('active', view === 'side');

        if (!plotDiv || !plotDiv.layout || !plotDiv.layout.scene) return;

        var camera = { eye: { x: -1.55, y: -1.55, z: 1.25 }, up: { x: 0, y: 0, z: 1 } };
        var proj = { type: 'perspective' };

        if (view === '2d') {
            camera = { eye: { x: 0.0001, y: 0.0001, z: 2.6 }, up: { x: 0, y: 1, z: 0 } };
            proj = { type: 'orthographic' };
        } else if (view === 'side') {
            camera = { eye: { x: -2.4, y: 0.0001, z: 0.35 }, up: { x: 0, y: 0, z: 1 } };
            proj = { type: 'perspective' };
        }

        Plotly.relayout(plotDiv, {
            'scene.camera': camera,
            'scene.camera.projection': proj
        });
    }

    function onSurfaceChange() {
        state.surfaceKey = document.getElementById('surface-select').value;
        var surf = SURFACES[state.surfaceKey];
        document.getElementById('insight-box').innerHTML = surf.insight;
        state.x0 = surf.defaultX0;
        state.y0 = surf.defaultY0;
        state.lr = surf.defaultLr;
        if (surf.defaultBeta !== undefined) state.beta = surf.defaultBeta;

        document.getElementById('x0-slider').min = surf.xRange[0];
        document.getElementById('x0-slider').max = surf.xRange[1];
        document.getElementById('x0-slider').value = state.x0;
        document.getElementById('x0-val').innerText = state.x0.toFixed(2);

        document.getElementById('y0-slider').min = surf.yRange[0];
        document.getElementById('y0-slider').max = surf.yRange[1];
        document.getElementById('y0-slider').value = state.y0;
        document.getElementById('y0-val').innerText = state.y0.toFixed(2);

        document.getElementById('lr-slider').value = state.lr;
        document.getElementById('lr-val').innerText = state.lr.toFixed(3);
        document.getElementById('beta-slider').value = state.beta;
        document.getElementById('beta-val').innerText = state.beta.toFixed(2);

        document.getElementById('hud-fstar').innerText = surf.fStar.toFixed(2);

        // Populate Presets
        var pContainer = document.getElementById('presets-container');
        pContainer.innerHTML = "";
        if (surf.presets) {
            surf.presets.forEach(function(p) {
                var btn = document.createElement('button');
                btn.className = 'btn-preset';
                btn.innerText = p.label;
                btn.onclick = function() {
                    state.x0 = p.x;
                    state.y0 = p.y;
                    document.getElementById('x0-slider').value = p.x;
                    document.getElementById('y0-slider').value = p.y;
                    document.getElementById('x0-val').innerText = p.x.toFixed(2);
                    document.getElementById('y0-val').innerText = p.y.toFixed(2);
                    recomputeSim(false);
                };
                pContainer.appendChild(btn);
            });
        }

        recomputeSim(true);
    }

    function updateHyperparams() {
        state.lr = parseFloat(document.getElementById('lr-slider').value);
        state.beta = parseFloat(document.getElementById('beta-slider').value);
        state.nSteps = parseInt(document.getElementById('steps-slider').value, 10);

        document.getElementById('lr-val').innerText = state.lr.toFixed(3);
        document.getElementById('beta-val').innerText = state.beta.toFixed(2);
        document.getElementById('steps-val').innerText = state.nSteps;

        document.getElementById('scrubber').max = state.nSteps;
        recomputeSim(false);
    }

    function updateCoordinates() {
        state.x0 = parseFloat(document.getElementById('x0-slider').value);
        state.y0 = parseFloat(document.getElementById('y0-slider').value);
        document.getElementById('x0-val').innerText = state.x0.toFixed(2);
        document.getElementById('y0-val').innerText = state.y0.toFixed(2);
        recomputeSim(false);
    }

    function randomizeStart() {
        var surf = SURFACES[state.surfaceKey];
        var rx = surf.xRange[0] + (0.15 + Math.random() * 0.7) * (surf.xRange[1] - surf.xRange[0]);
        var ry = surf.yRange[0] + (0.15 + Math.random() * 0.7) * (surf.yRange[1] - surf.yRange[0]);
        state.x0 = parseFloat(rx.toFixed(2));
        state.y0 = parseFloat(ry.toFixed(2));
        document.getElementById('x0-slider').value = state.x0;
        document.getElementById('y0-slider').value = state.y0;
        document.getElementById('x0-val').innerText = state.x0.toFixed(2);
        document.getElementById('y0-val').innerText = state.y0.toFixed(2);
        recomputeSim(false);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Plotly WebGL Viewport Renderer (No-Lag Architecture)
    // ═══════════════════════════════════════════════════════════════════
    var plotDiv = document.getElementById('plotly-canvas');
    var isInitialized = false;

    function recomputeSim(redrawFullScene) {
        var surf = SURFACES[state.surfaceKey];
        var t0 = performance.now();

        if (state.mode === 'race') {
            state.simData = {};
            for (var name in OPT_CONFIG) {
                if (OPT_CONFIG[name].active) {
                    state.simData[name] = runOptimizer(
                        name, surf, state.x0, state.y0, state.nSteps, state.lr, state.beta
                    );
                }
            }
        } else {
            state.riverData = runRiverSimulation(surf, 26, state.nSteps, state.lr);
        }

        var totalLatency = performance.now() - t0;
        var activeCount = Object.keys(state.simData).length || 1;
        var perStepUs = (totalLatency * 1000.0) / (state.nSteps * activeCount);
        document.getElementById('hud-latency').innerText = perStepUs.toFixed(1) + " µs";

        state.currentStep = state.nSteps; // Start with full path pre-rendered
        updateTelemetryHUD();
        renderConvergenceChart();

        if (redrawFullScene || !isInitialized) {
            renderFullScene();
        } else {
            updateDynamicTracesOnly();
        }
    }

    function renderFullScene() {
        var surf = SURFACES[state.surfaceKey];
        var mesh = generateMesh(surf, 42);

        var traces = [];

        // 1. Sleek Shaded 3D Surface (Zero mesh wireframe lines, clean projected floor contours)
        traces.push({
            type: 'surface',
            x: mesh.X,
            y: mesh.Y,
            z: mesh.Z,
            colorscale: 'Plasma',
            opacity: 0.92,
            showscale: false,
            lighting: {
                ambient: 0.80,
                diffuse: 0.85,
                specular: 0.35,
                roughness: 0.50,
                fresnel: 0.25
            },
            contours: {
                x: { show: false },
                y: { show: false },
                z: {
                    show: true,
                    usecolormap: true,
                    project: { z: true }, // Projected strictly onto bottom plane
                    width: 1.5
                }
            },
            hoverinfo: 'none',
            name: 'Loss Topography'
        });

        // 2. Unmistakable Global Minimum Target Beacon
        var zStar = surf.renderZ(surf.f(surf.fStarX, surf.fStarY)) + 0.15;
        traces.push({
            type: 'scatter3d',
            x: [surf.fStarX],
            y: [surf.fStarY],
            z: [zStar],
            mode: 'markers+text',
            name: 'Global Minimum θ*',
            marker: { size: 10, color: '#FBBF24', symbol: 'diamond', line: { color: '#FFFFFF', width: 2.5 } },
            text: ['🎯 Global Min (f*=' + surf.fStar.toFixed(2) + ')'],
            textposition: 'top center',
            textfont: { color: '#FBBF24', size: 11, family: 'sans-serif' },
            hoverinfo: 'text'
        });

        // 3. Starting Position Landmark
        var z0 = surf.renderZ(surf.f(state.x0, state.y0)) + 0.15;
        traces.push({
            type: 'scatter3d',
            x: [state.x0],
            y: [state.y0],
            z: [z0],
            mode: 'markers+text',
            name: 'θ₀ Start',
            marker: { size: 9, color: '#EF4444', symbol: 'circle', line: { color: '#FFFFFF', width: 2 } },
            text: ['📍 Start θ₀'],
            textposition: 'top center',
            textfont: { color: '#EF4444', size: 10 }
        });

        // 4. Saddle Landmark (if surface has saddle)
        if (surf.saddlePoint) {
            var zSaddle = surf.renderZ(surf.f(surf.saddlePoint.x, surf.saddlePoint.y)) + 0.15;
            traces.push({
                type: 'scatter3d',
                x: [surf.saddlePoint.x],
                y: [surf.saddlePoint.y],
                z: [zSaddle],
                mode: 'markers+text',
                name: 'Saddle Point',
                marker: { size: 9, color: '#F87171', symbol: 'cross', line: { color: '#FFFFFF', width: 2.5 } },
                text: [surf.saddlePoint.label],
                textposition: 'top center',
                textfont: { color: '#F87171', size: 10 }
            });
        }

        var stepIdx = Math.min(state.currentStep, state.nSteps);

        if (state.mode === 'race') {
            for (var name in state.simData) {
                var sim = state.simData[name];
                var color = OPT_CONFIG[name].color;

                var sliceX = sim.xs.slice(0, stepIdx + 1);
                var sliceY = sim.ys.slice(0, stepIdx + 1);
                var sliceZ = sim.renderZs.slice(0, stepIdx + 1).map(function(v) { return v + 0.10; });

                // Trajectory Ribbon
                traces.push({
                    type: 'scatter3d',
                    x: sliceX,
                    y: sliceY,
                    z: sliceZ,
                    mode: 'lines',
                    name: name,
                    line: { color: color, width: 6 }
                });

                // Rolling Sphere Head
                var hx = sliceX[sliceX.length - 1], hy = sliceY[sliceY.length - 1], hz = sliceZ[sliceZ.length - 1];
                traces.push({
                    type: 'scatter3d',
                    x: [hx],
                    y: [hy],
                    z: [hz],
                    mode: 'markers',
                    showlegend: false,
                    marker: { size: 7.5, color: color, symbol: 'circle', line: { color: '#FFFFFF', width: 1.5 } }
                });
            }
        } else {
            // River mode: Glowing liquid streams
            var pal = ['#38BDF8', '#06B6D4', '#2DD4BF', '#34D399', '#60A5FA', '#818CF8'];
            for (var i = 0; i < state.riverData.length; i++) {
                var st = state.riverData[i];
                var color = pal[i % pal.length];
                traces.push({
                    type: 'scatter3d',
                    x: st.xs.slice(0, stepIdx + 1),
                    y: st.ys.slice(0, stepIdx + 1),
                    z: st.renderZs.slice(0, stepIdx + 1).map(function(v) { return v + 0.06; }),
                    mode: 'lines',
                    showlegend: false,
                    line: { color: color, width: 3.5 }
                });
            }
        }

        var axisCfg = {
            backgroundcolor: '#111827',
            gridcolor: '#1F2937',
            showbackground: true,
            tickfont: { color: '#6B7280', size: 9 },
            titlefont: { color: '#9CA3AF', size: 11 }
        };

        var camera = { eye: { x: -1.55, y: -1.55, z: 1.25 }, up: { x: 0, y: 0, z: 1 } };
        var proj = { type: 'perspective' };

        if (state.viewMode === '2d') {
            camera = { eye: { x: 0.0001, y: 0.0001, z: 2.6 }, up: { x: 0, y: 1, z: 0 } };
            proj = { type: 'orthographic' };
        } else if (state.viewMode === 'side') {
            camera = { eye: { x: -2.4, y: 0.0001, z: 0.35 }, up: { x: 0, y: 0, z: 1 } };
            proj = { type: 'perspective' };
        }

        var layout = {
            paper_bgcolor: '#0B0F19',
            plot_bgcolor: '#111827',
            margin: { l: 0, r: 0, t: 10, b: 0 },
            showlegend: false,
            scene: {
                xaxis: Object.assign({}, axisCfg, { title: 'Parameter θ₁' }),
                yaxis: Object.assign({}, axisCfg, { title: 'Parameter θ₂' }),
                zaxis: Object.assign({}, axisCfg, { title: 'Loss Topography L(θ)' }),
                bgcolor: '#0B0F19',
                camera: camera,
                projection: proj
            }
        };

        var config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.react(plotDiv, traces, layout, config).then(function() {
            if (!isInitialized) {
                isInitialized = true;
                document.getElementById('loading-screen').style.display = 'none';

                // Click-to-drop start position on 3D surface
                plotDiv.on('plotly_click', function(data) {
                    if (data && data.points && data.points[0]) {
                        var pt = data.points[0];
                        if (pt.x !== undefined && pt.y !== undefined) {
                            state.x0 = parseFloat(pt.x.toFixed(2));
                            state.y0 = parseFloat(pt.y.toFixed(2));
                            document.getElementById('x0-slider').value = state.x0;
                            document.getElementById('y0-slider').value = state.y0;
                            document.getElementById('x0-val').innerText = state.x0.toFixed(2);
                            document.getElementById('y0-val').innerText = state.y0.toFixed(2);
                            recomputeSim(false);
                        }
                    }
                });
            }
        });
    }

    // High-performance trace restyler: updates lines & markers at 60 FPS WITHOUT re-triangulating the 3D surface!
    function updateDynamicTracesOnly() {
        if (!isInitialized || !plotDiv.data) return;

        var surf = SURFACES[state.surfaceKey];
        var stepIdx = Math.min(state.currentStep, state.nSteps);
        var startOffset = surf.saddlePoint ? 4 : 3;

        // Update Start Marker
        var z0 = surf.renderZ(surf.f(state.x0, state.y0)) + 0.15;
        Plotly.restyle(plotDiv, { x: [[state.x0]], y: [[state.y0]], z: [[z0]] }, [2]);

        var updateX = [], updateY = [], updateZ = [];
        var traceIndices = [];

        if (state.mode === 'race') {
            var idx = startOffset;
            for (var name in state.simData) {
                if (idx >= plotDiv.data.length) break;
                var sim = state.simData[name];
                var sliceX = sim.xs.slice(0, stepIdx + 1);
                var sliceY = sim.ys.slice(0, stepIdx + 1);
                var sliceZ = sim.renderZs.slice(0, stepIdx + 1).map(function(v) { return v + 0.10; });

                // Ribbon
                updateX.push(sliceX); updateY.push(sliceY); updateZ.push(sliceZ);
                traceIndices.push(idx);
                idx++;

                // Rolling Sphere Head
                if (idx < plotDiv.data.length) {
                    var hx = sliceX[sliceX.length - 1], hy = sliceY[sliceY.length - 1], hz = sliceZ[sliceZ.length - 1];
                    updateX.push([hx]); updateY.push([hy]); updateZ.push([hz]);
                    traceIndices.push(idx);
                    idx++;
                }
            }
        } else {
            var idx2 = startOffset;
            for (var i = 0; i < state.riverData.length; i++) {
                if (idx2 >= plotDiv.data.length) break;
                var st = state.riverData[i];
                var sliceX2 = st.xs.slice(0, stepIdx + 1);
                var sliceY2 = st.ys.slice(0, stepIdx + 1);
                var sliceZ2 = st.renderZs.slice(0, stepIdx + 1).map(function(v) { return v + 0.06; });

                updateX.push(sliceX2); updateY.push(sliceY2); updateZ.push(sliceZ2);
                traceIndices.push(idx2);
                idx2++;
            }
        }

        if (traceIndices.length > 0) {
            Plotly.restyle(plotDiv, { x: updateX, y: updateY, z: updateZ }, traceIndices);
        }
    }

    function updateTelemetryHUD() {
        var tbody = document.getElementById('telemetry-tbody');
        tbody.innerHTML = "";
        var surf = SURFACES[state.surfaceKey];
        var stepIdx = Math.min(state.currentStep, state.nSteps);

        if (state.mode === 'race') {
            for (var name in state.simData) {
                var sim = state.simData[name];
                var color = OPT_CONFIG[name].color;
                var curLoss = sim.losses[Math.min(stepIdx, sim.losses.length - 1)];
                var curX = sim.xs[Math.min(stepIdx, sim.xs.length - 1)];
                var curY = sim.ys[Math.min(stepIdx, sim.ys.length - 1)];
                var distToMin = Math.hypot(curX - surf.fStarX, curY - surf.fStarY);

                var tr = document.createElement('tr');

                var statusBadge = "";
                if (sim.epsStep !== null && stepIdx >= sim.epsStep) {
                    statusBadge = '<span style="color:#34D399;font-weight:700;">★ Conv (' + sim.epsStep + ' st)</span>';
                } else if (distToMin < 0.25) {
                    statusBadge = '<span style="color:#60A5FA;">Near Basin</span>';
                } else if (sim.pathEff < 0.25) {
                    statusBadge = '<span style="color:#FBBF24;">⚡ Oscillating</span>';
                } else {
                    statusBadge = '<span style="color:#9CA3AF;">Descending</span>';
                }

                tr.innerHTML =
                    '<td><span style="color:' + color + ';font-weight:700;">● ' + name + '</span></td>' +
                    '<td>' + (Math.abs(curLoss) > 1e4 ? curLoss.toExponential(1) : curLoss.toFixed(4)) + '</td>' +
                    '<td>' + distToMin.toFixed(3) + '</td>' +
                    '<td>' + sim.pathEff.toFixed(3) + '</td>' +
                    '<td>' + statusBadge + '</td>';

                tbody.appendChild(tr);
            }
        } else {
            var tr = document.createElement('tr');
            var curDrop = 0;
            if (state.riverData.length > 0) {
                var l0 = state.riverData[0].losses[0];
                var lt = state.riverData[0].losses[Math.min(stepIdx, state.riverData[0].losses.length - 1)];
                curDrop = Math.max(0, l0 - lt);
            }
            tr.innerHTML = '<td colspan="5" style="color:#38BDF8;text-align:center;padding:8px 0;">🌊 26 liquid tributaries cascading along steepest descent into the global minimum basin (Potential drop: ' + curDrop.toFixed(2) + ')</td>';
            tbody.appendChild(tr);
        }

        document.getElementById('step-counter').innerText = 'Step: ' + stepIdx + ' / ' + state.nSteps;
        document.getElementById('scrubber').value = stepIdx;
        document.getElementById('chart-step-lbl').innerText = 'Step ' + stepIdx + ' / ' + state.nSteps;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Synchronized Convergence Mini-Chart (Log-Loss vs Step)
    // ═══════════════════════════════════════════════════════════════════
    function renderConvergenceChart() {
        var svg = document.getElementById('convergence-svg');
        var stepIdx = Math.min(state.currentStep, state.nSteps);
        var surf = SURFACES[state.surfaceKey];

        if (state.mode !== 'race' || Object.keys(state.simData).length === 0) {
            svg.innerHTML = '<text x="225" y="35" fill="#6B7280" font-size="11" text-anchor="middle">Fluid Potential Flow Active</text>';
            return;
        }

        var W = 450, H = 60;
        var padding = { l: 30, r: 15, t: 8, b: 12 };
        var plotW = W - padding.l - padding.r;
        var plotH = H - padding.t - padding.b;

        // Compute global min and max log loss across active optimizers
        var minLog = 999, maxLog = -999;
        for (var name in state.simData) {
            var sim = state.simData[name];
            for (var t = 0; t <= state.nSteps; t++) {
                var l = sim.losses[Math.min(t, sim.losses.length - 1)];
                var diff = Math.max(1e-5, l - surf.fStar);
                var logVal = Math.log10(diff);
                if (logVal < minLog) minLog = logVal;
                if (logVal > maxLog) maxLog = logVal;
            }
        }
        if (maxLog - minLog < 1.0) maxLog = minLog + 1.0;

        var html = '';
        // Subtle grid lines
        html += '<line x1="' + padding.l + '" y1="' + padding.t + '" x2="' + (W - padding.r) + '" y2="' + padding.t + '" stroke="#1F2937" stroke-width="1"/>';
        html += '<line x1="' + padding.l + '" y1="' + (H - padding.b) + '" x2="' + (W - padding.r) + '" y2="' + (H - padding.b) + '" stroke="#1F2937" stroke-width="1"/>';

        // Draw curves up to current step
        for (var name in state.simData) {
            var sim = state.simData[name];
            var color = OPT_CONFIG[name].color;
            var points = [];

            for (var t = 0; t <= stepIdx; t++) {
                var l = sim.losses[Math.min(t, sim.losses.length - 1)];
                var diff = Math.max(1e-5, l - surf.fStar);
                var logVal = Math.log10(diff);
                var normX = t / state.nSteps;
                var normY = (logVal - minLog) / (maxLog - minLog);
                var xPx = padding.l + normX * plotW;
                var yPx = (H - padding.b) - normY * plotH;
                points.push(xPx.toFixed(1) + ',' + yPx.toFixed(1));
            }

            if (points.length > 1) {
                html += '<polyline fill="none" stroke="' + color + '" stroke-width="2" points="' + points.join(' ') + '"/>';
                var lastPt = points[points.length - 1].split(',');
                html += '<circle cx="' + lastPt[0] + '" cy="' + lastPt[1] + '" r="3" fill="' + color + '"/>';
            }
        }

        // Vertical scrubber indicator
        var scrubX = padding.l + (stepIdx / state.nSteps) * plotW;
        html += '<line x1="' + scrubX + '" y1="' + padding.t + '" x2="' + scrubX + '" y2="' + (H - padding.b) + '" stroke="#60A5FA" stroke-width="1.5" stroke-dasharray="2,2"/>';

        svg.innerHTML = html;
    }

    // ═══════════════════════════════════════════════════════════════════
    // 60 FPS Butter-Smooth Animation Loop
    // ═══════════════════════════════════════════════════════════════════
    function togglePlay() {
        if (state.isPlaying) {
            pauseSim();
        } else {
            playSim();
        }
    }

    function playSim() {
        state.isPlaying = true;
        document.getElementById('play-icon').innerText = "⏸";
        document.getElementById('play-text').innerText = "Pause";
        if (state.currentStep >= state.nSteps) {
            state.currentStep = 0;
        }

        state.lastFrameTime = performance.now();
        requestAnimationFrame(animStep);
    }

    function animStep(timestamp) {
        if (!state.isPlaying) return;

        var interval = Math.max(16, 40 / state.speed);
        if (timestamp - state.lastFrameTime >= interval) {
            state.lastFrameTime = timestamp;
            if (state.currentStep < state.nSteps) {
                state.currentStep++;
                updateTelemetryHUD();
                renderConvergenceChart();
                updateDynamicTracesOnly();
            } else {
                pauseSim();
                return;
            }
        }
        requestAnimationFrame(animStep);
    }

    function pauseSim() {
        state.isPlaying = false;
        document.getElementById('play-icon').innerText = "▶";
        document.getElementById('play-text').innerText = (state.mode === 'race' ? 'Play Race' : 'Flow River');
    }

    function resetSim() {
        pauseSim();
        state.currentStep = 0;
        updateTelemetryHUD();
        renderConvergenceChart();
        updateDynamicTracesOnly();
    }

    function stepForward() {
        pauseSim();
        if (state.currentStep < state.nSteps) {
            state.currentStep = Math.min(state.currentStep + 5, state.nSteps);
            updateTelemetryHUD();
            renderConvergenceChart();
            updateDynamicTracesOnly();
        }
    }

    function onScrub(val) {
        pauseSim();
        state.currentStep = parseInt(val, 10);
        updateTelemetryHUD();
        renderConvergenceChart();
        updateDynamicTracesOnly();
    }

    function setSpeed(spd) {
        state.speed = spd;
        ['spd-0.5', 'spd-1', 'spd-2', 'spd-4'].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) el.style.background = "#1F2937";
        });
        var activeEl = document.getElementById('spd-' + spd);
        if (activeEl) activeEl.style.background = "#2563EB";
    }

    // ═══════════════════════════════════════════════════════════════════
    // Initialization & Resilient CDN Script Loader
    // ═══════════════════════════════════════════════════════════════════
    function bootstrap() {
        buildOptimizerChips();
        onSurfaceChange();
    }

    (function loadPlotly() {
        if (typeof Plotly !== 'undefined') {
            bootstrap();
        } else {
            var s = document.createElement('script');
            s.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
            s.crossOrigin = 'anonymous';
            s.onload = bootstrap;
            s.onerror = function() {
                var s2 = document.createElement('script');
                s2.src = 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js';
                s2.onload = bootstrap;
                s2.onerror = function() {
                    var loader = document.getElementById('loading-screen');
                    loader.innerHTML = '<div style="color:#EF4444;font-size:16px;font-weight:700;">Plotly CDN Load Error</div>' +
                                       '<div style="color:#9CA3AF;font-size:12px;margin-top:6px;">Check internet connectivity or adblocker restrictions.</div>';
                };
                document.head.appendChild(s2);
            };
            document.head.appendChild(s);
        }
    })();
    </script>
</body>
</html>
"""


def export_interactive_html():
    print("  🌐 Compiling educational WebGL 3D physics laboratory into assets/interactive_3d_landscape.html ...")
    output_path = os.path.join(ASSETS_DIR, "interactive_3d_landscape.html")

    html_content = build_interactive_app_html()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"    ✓ Successfully compiled {output_path} ({len(html_content) / 1024:.1f} KB)")


if __name__ == "__main__":
    export_interactive_html()
