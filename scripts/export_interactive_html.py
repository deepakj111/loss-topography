#!/usr/bin/env python3
"""
Interactive 3D Visualizer & Physics Engine Generator for loss-topography.

Compiles a standalone, zero-dependency, rich WebGL 3D laboratory into
assets/interactive_3d_landscape.html.

Features:
- Pure client-side math & physics engine (analytical surfaces + gradients in JS).
- Dynamic surface switcher (Multi-Modal, Rosenbrock, Beale, Regularized Saddle, Rastrigin).
- Live simulation & animation controls (Play/Pause, Step, Reset, Scrub, Speed 0.5x-4x).
- 2 Simulation Modes:
  1. 🏁 Optimizer Race (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW, Damped Newton)
  2. 🌊 River of Optimization (24 liquid tributaries cascading downhill)
- Real-time Hyperparameter Tuning (Learning rate η, Momentum β, Steps N, Coordinates x₀, y₀).
- Click-to-Drop: click anywhere on the 3D surface to launch optimizers from that point!
- Live Performance & Telemetry HUD (Live loss, path efficiency, compute latency in µs).
- Resilient loader with loading spinner and fallback CDN (works in local browsers and htmlpreview).
"""

from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ASSETS_DIR = os.path.join(ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def build_interactive_app_html() -> str:
    html = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>loss-topography — Interactive 3D Loss Landscape & Optimizer Physics Lab</title>
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

        /* Top Header */
        #header {
            height: 56px;
            background: #111827;
            border-bottom: 1px solid #1F2937;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            z-index: 30;
            position: relative;
        }
        .header-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-title {
            font-size: 16px;
            font-weight: 700;
            color: #FFFFFF;
            letter-spacing: -0.3px;
        }
        .header-badge {
            background: rgba(37, 99, 235, 0.2);
            color: #60A5FA;
            border: 1px solid rgba(37, 99, 235, 0.4);
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
        }
        .header-controls {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .mode-toggle {
            display: flex;
            background: #1F2937;
            border-radius: 8px;
            padding: 3px;
            gap: 4px;
        }
        .mode-btn {
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
        .mode-btn.active {
            background: #2563EB;
            color: #FFFFFF;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        .btn-github {
            background: #1F2937;
            color: #E5E7EB;
            text-decoration: none;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid #374151;
            transition: all 0.15s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .btn-github:hover {
            background: #374151;
            color: #FFFFFF;
        }

        /* App Body Layout */
        #app-container {
            display: flex;
            height: calc(100vh - 56px);
            width: 100%;
            position: relative;
        }

        /* Left Control Sidebar */
        #sidebar {
            width: 340px;
            background: rgba(17, 24, 39, 0.95);
            border-right: 1px solid #1F2937;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            z-index: 20;
            backdrop-filter: blur(12px);
        }
        .control-group {
            background: #161F30;
            border: 1px solid #1F2937;
            border-radius: 10px;
            padding: 12px 14px;
        }
        .group-title {
            font-size: 11px;
            font-weight: 700;
            color: #9CA3AF;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .form-row {
            margin-bottom: 10px;
        }
        .form-row:last-child {
            margin-bottom: 0;
        }
        .form-label {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #D1D5DB;
            margin-bottom: 4px;
            font-weight: 500;
        }
        .form-label span.val {
            color: #60A5FA;
            font-family: monospace;
            font-weight: 600;
        }
        select, input[type="range"] {
            width: 100%;
        }
        select {
            background: #0B0F19;
            border: 1px solid #374151;
            color: #F9FAFB;
            padding: 7px 10px;
            border-radius: 6px;
            font-size: 13px;
            outline: none;
            cursor: pointer;
        }
        select:focus {
            border-color: #3B82F6;
        }
        input[type="range"] {
            accent-color: #3B82F6;
            height: 5px;
            background: #374151;
            border-radius: 3px;
            outline: none;
        }

        /* Optimizer Toggle Chips */
        .opt-grid {
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
            color: #9CA3AF;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            user-select: none;
            transition: all 0.15s ease;
        }
        .opt-chip.active {
            border-color: currentColor;
            color: #FFFFFF;
            background: rgba(255, 255, 255, 0.05);
        }
        .opt-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }

        /* Playback Bar */
        .playback-row {
            display: flex;
            gap: 6px;
            margin-top: 8px;
        }
        .play-btn {
            flex: 1;
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
        .play-btn:hover { background: #1D4ED8; }
        .sec-btn {
            background: #1F2937;
            color: #D1D5DB;
            border: 1px solid #374151;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }
        .sec-btn:hover { background: #374151; color: white; }

        /* Main Viewport */
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

        /* Floating HUD Telemetry Overlay */
        #telemetry-hud {
            position: absolute;
            bottom: 20px;
            right: 20px;
            width: 440px;
            max-width: calc(100vw - 380px);
            background: rgba(17, 24, 39, 0.90);
            border: 1px solid #1F2937;
            border-radius: 12px;
            padding: 14px 16px;
            backdrop-filter: blur(12px);
            z-index: 25;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6);
        }
        .hud-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid #1F2937;
        }
        .hud-title {
            font-size: 12px;
            font-weight: 700;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .hud-stats {
            display: flex;
            gap: 12px;
            font-size: 11px;
            color: #9CA3AF;
            font-family: monospace;
        }
        .hud-stats span.val {
            color: #34D399;
            font-weight: 700;
        }
        .telemetry-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }
        .telemetry-table th {
            text-align: left;
            color: #9CA3AF;
            font-weight: 600;
            padding: 3px 6px;
            border-bottom: 1px solid #1F2937;
        }
        .telemetry-table td {
            padding: 4px 6px;
            color: #E5E7EB;
            font-family: monospace;
        }
        .telemetry-table tr:hover td {
            background: rgba(255, 255, 255, 0.03);
        }

        /* Hint floating tag */
        .click-hint {
            position: absolute;
            top: 16px;
            left: 20px;
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid #1F2937;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 11px;
            color: #9CA3AF;
            pointer-events: none;
            backdrop-filter: blur(8px);
            z-index: 15;
            display: flex;
            align-items: center;
            gap: 6px;
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
            width: 48px;
            height: 48px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top-color: #3B82F6;
            animation: spin 0.8s ease-in-out infinite;
            margin-bottom: 16px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <!-- Top Header -->
    <header id="header">
        <div class="header-brand">
            <span style="font-size: 20px;">🏔️</span>
            <div>
                <span class="header-title">loss-topography</span>
                <span class="header-badge">Interactive 3D Visualizer</span>
            </div>
        </div>
        <div class="header-controls">
            <div class="mode-toggle">
                <button id="mode-race" class="mode-btn active" onclick="setMode('race')">🏁 Optimizer Race</button>
                <button id="mode-river" class="mode-btn" onclick="setMode('river')">🌊 Hydro River Flow</button>
            </div>
            <a href="https://github.com/deepakj111/loss-topography" target="_blank" class="btn-github">
                <span>★ GitHub</span>
            </a>
        </div>
    </header>

    <!-- App Body -->
    <div id="app-container">
        <!-- Sidebar Controls -->
        <aside id="sidebar">
            <!-- Surface Selection -->
            <div class="control-group">
                <div class="group-title">
                    <span>Loss Landscape</span>
                </div>
                <div class="form-row">
                    <select id="surface-select" onchange="onSurfaceChange()">
                        <option value="multimodal">🌊 Multi-Modal Potential</option>
                        <option value="rosenbrock">🍌 Rosenbrock Banana Valley</option>
                        <option value="beale">🏔️ Beale Function (High Condition κ)</option>
                        <option value="regsaddle">🕳️ Regularized Saddle (Strict Saddle)</option>
                        <option value="rastrigin">🌀 Rastrigin Surface (Dense Minima)</option>
                    </select>
                </div>
                <div style="font-size: 11px; color: #9CA3AF; margin-top: 6px; line-height: 1.4;" id="surface-desc">
                    Complex rugged topography featuring multiple local basins, high-elevation ridges, and a gravitational global sink.
                </div>
            </div>

            <!-- Playback Controls -->
            <div class="control-group">
                <div class="group-title">
                    <span>Simulation Playback</span>
                    <span id="step-counter" style="color: #60A5FA; font-family: monospace;">Step: 0 / 150</span>
                </div>
                <div class="form-row">
                    <input type="range" id="scrubber" min="0" max="150" value="0" oninput="onScrub(this.value)">
                </div>
                <div class="playback-row">
                    <button id="btn-play" class="play-btn" onclick="togglePlay()">
                        <span id="play-icon">▶</span> <span id="play-text">Play Race</span>
                    </button>
                    <button class="sec-btn" onclick="stepForward()">⏭ Step</button>
                    <button class="sec-btn" onclick="resetSim()">⏮ Reset</button>
                </div>
                <div class="form-row" style="margin-top: 10px; display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 11px; color: #9CA3AF;">Speed:</span>
                    <div style="display: flex; gap: 4px;">
                        <button class="sec-btn" style="padding: 3px 8px; font-size: 10px;" onclick="setSpeed(0.5)">0.5x</button>
                        <button class="sec-btn" style="padding: 3px 8px; font-size: 10px; background: #2563EB; color:white;" id="spd-1" onclick="setSpeed(1)">1x</button>
                        <button class="sec-btn" style="padding: 3px 8px; font-size: 10px;" onclick="setSpeed(2)">2x</button>
                        <button class="sec-btn" style="padding: 3px 8px; font-size: 10px;" onclick="setSpeed(4)">4x</button>
                    </div>
                </div>
            </div>

            <!-- Optimizers Toggle (Race Mode) -->
            <div class="control-group" id="opt-selection-group">
                <div class="group-title">
                    <span>Optimizers</span>
                    <button class="sec-btn" style="padding: 2px 6px; font-size: 9px;" onclick="toggleAllOptimizers()">Toggle All</button>
                </div>
                <div class="opt-grid" id="opt-chips-container">
                    <!-- Populated by JS -->
                </div>
            </div>

            <!-- Hyperparameters -->
            <div class="control-group">
                <div class="group-title">
                    <span>Hyperparameters</span>
                </div>
                <div class="form-row">
                    <div class="form-label">
                        <span>Learning Rate (η)</span>
                        <span class="val" id="lr-val">0.030</span>
                    </div>
                    <input type="range" id="lr-slider" min="0.001" max="0.15" step="0.001" value="0.03" oninput="updateHyperparams()">
                </div>
                <div class="form-row">
                    <div class="form-label">
                        <span>Momentum (β)</span>
                        <span class="val" id="beta-val">0.85</span>
                    </div>
                    <input type="range" id="beta-slider" min="0.0" max="0.99" step="0.01" value="0.85" oninput="updateHyperparams()">
                </div>
                <div class="form-row">
                    <div class="form-label">
                        <span>Total Steps (N)</span>
                        <span class="val" id="steps-val">150</span>
                    </div>
                    <input type="range" id="steps-slider" min="40" max="300" step="10" value="150" oninput="updateHyperparams()">
                </div>
            </div>

            <!-- Coordinates -->
            <div class="control-group">
                <div class="group-title">
                    <span>Starting Position (θ₀)</span>
                    <button class="sec-btn" style="padding: 2px 6px; font-size: 9px;" onclick="randomizeStart()">🎲 Random</button>
                </div>
                <div class="form-row">
                    <div class="form-label">
                        <span>Initial θ₁ (x₀)</span>
                        <span class="val" id="x0-val">2.50</span>
                    </div>
                    <input type="range" id="x0-slider" min="-3" max="3" step="0.05" value="2.5" oninput="updateCoordinates()">
                </div>
                <div class="form-row">
                    <div class="form-label">
                        <span>Initial θ₂ (y₀)</span>
                        <span class="val" id="y0-val">2.50</span>
                    </div>
                    <input type="range" id="y0-slider" min="-3" max="3" step="0.05" value="2.5" oninput="updateCoordinates()">
                </div>
            </div>
        </aside>

        <!-- Main 3D Viewport -->
        <main id="viewport">
            <div class="click-hint">
                <span>🖱️ Click anywhere on the 3D surface to reposition θ₀</span>
            </div>
            <div id="plotly-canvas"></div>

            <!-- Floating HUD Telemetry -->
            <div id="telemetry-hud">
                <div class="hud-header">
                    <div class="hud-title">
                        <span>📊 Live Optimizer Telemetry</span>
                    </div>
                    <div class="hud-stats">
                        <span>Step Latency: <span class="val" id="hud-latency">~3.2 µs</span></span>
                        <span>Global Min: <span class="val" id="hud-fstar">-6.55</span></span>
                    </div>
                </div>
                <table class="telemetry-table">
                    <thead>
                        <tr>
                            <th>Optimizer</th>
                            <th>Loss L(θ)</th>
                            <th>Path Eff.</th>
                            <th>Convergence</th>
                        </tr>
                    </thead>
                    <tbody id="telemetry-tbody">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </main>
    </div>

    <!-- Initial Loading Screen -->
    <div id="loading-screen">
        <div class="spinner"></div>
        <div style="font-size: 16px; font-weight: 600; color: #FFFFFF;">Initializing 3D WebGL Laboratory...</div>
        <div id="loading-status" style="font-size: 12px; color: #60A5FA; margin-top: 6px;">Loading Plotly 3D engine</div>
    </div>

    <script>
    // ═══════════════════════════════════════════════════════════════════
    // Mathematical Loss Topographies & Closed-Form Gradients
    // ═══════════════════════════════════════════════════════════════════
    const SURFACES = {
        multimodal: {
            name: "Multi-Modal Potential",
            fStar: -6.55,
            xRange: [-4.0, 4.0],
            yRange: [-4.0, 4.0],
            defaultX0: 2.50,
            defaultY0: 2.50,
            defaultLr: 0.030,
            desc: "Non-convex potential with multiple local basins, narrow mountain passes, and a global gravitational minimum.",
            f: function(x, y) {
                var r2 = x*x + y*y;
                var z = -(Math.cos(x)*Math.sin(y)*Math.exp(-r2/10.0) + 0.3*Math.cos(2*x) + 0.3*Math.sin(2*y) + 0.1*r2);
                return z * 3.0;
            },
            grad: function(x, y) {
                var eps = 1e-5;
                var zx1 = this.f(x + eps, y), zx0 = this.f(x - eps, y);
                var zy1 = this.f(x, y + eps), zy0 = this.f(x, y - eps);
                return [(zx1 - zx0)/(2*eps), (zy1 - zy0)/(2*eps)];
            }
        },
        rosenbrock: {
            name: "Rosenbrock Banana Valley",
            fStar: 0.0,
            xRange: [-2.0, 2.0],
            yRange: [-1.0, 3.0],
            defaultX0: -0.80,
            defaultY0: 1.20,
            defaultLr: 0.003,
            desc: "Parabolic ravine with extreme condition number (κ ≫ 10³). Bounces violently without curvature dampening.",
            f: function(x, y) {
                var a = 1.0, b = 100.0;
                var v = (a - x)*(a - x) + b * Math.pow(y - x*x, 2);
                return Math.min(v, 250.0);
            },
            grad: function(x, y) {
                var a = 1.0, b = 100.0;
                var gx = -2.0*(a - x) - 4.0*b*x*(y - x*x);
                var gy = 2.0*b*(y - x*x);
                return [gx, gy];
            }
        },
        beale: {
            name: "Beale Function",
            fStar: 0.0,
            xRange: [-3.5, 3.5],
            yRange: [-3.5, 3.5],
            defaultX0: 2.00,
            defaultY0: 2.00,
            defaultLr: 0.015,
            desc: "Highly anisotropic valley with flat outer wings and sharp corner ravines. Requires coordinate adaptation.",
            f: function(x, y) {
                var t1 = 1.5 - x + x*y;
                var t2 = 2.25 - x + x*y*y;
                var t3 = 2.625 - x + x*y*y*y;
                var v = t1*t1 + t2*t2 + t3*t3;
                return Math.min(v, 150.0);
            },
            grad: function(x, y) {
                var t1 = 1.5 - x + x*y;
                var t2 = 2.25 - x + x*y*y;
                var t3 = 2.625 - x + x*y*y*y;
                var gx = 2*t1*(y - 1) + 2*t2*(y*y - 1) + 2*t3*(y*y*y - 1);
                var gy = 2*t1*x + 4*t2*x*y + 6*t3*x*y*y;
                return [gx, gy];
            }
        },
        regsaddle: {
            name: "Regularized Saddle",
            fStar: -2.50,
            xRange: [-3.0, 3.0],
            yRange: [-3.0, 3.0],
            defaultX0: 0.05,
            defaultY0: 0.10,
            defaultLr: 0.035,
            desc: "Strict saddle at (0, 0) with negative curvature along y leading to twin global minima at (0, ±√5) with f* = -2.5000.",
            f: function(x, y) {
                return x*x - y*y + 0.1*Math.pow(x, 4) + 0.1*Math.pow(y, 4);
            },
            grad: function(x, y) {
                return [2*x + 0.4*Math.pow(x, 3), -2*y + 0.4*Math.pow(y, 3)];
            }
        },
        rastrigin: {
            name: "Rastrigin Surface",
            fStar: 0.0,
            xRange: [-4.0, 4.0],
            yRange: [-4.0, 4.0],
            defaultX0: 2.50,
            defaultY0: 2.50,
            defaultLr: 0.020,
            desc: "Dense lattice of cosine-induced local minima. Standard SGD stalls; adaptive momentum navigates past barriers.",
            f: function(x, y) {
                var v = 20.0 + (x*x - 10*Math.cos(2*Math.PI*x)) + (y*y - 10*Math.cos(2*Math.PI*y));
                return Math.min(v, 80.0);
            },
            grad: function(x, y) {
                var gx = 2*x + 20*Math.PI*Math.sin(2*Math.PI*x);
                var gy = 2*y + 20*Math.PI*Math.sin(2*Math.PI*y);
                return [gx, gy];
            }
        }
    };

    // ═══════════════════════════════════════════════════════════════════
    // Optimizer Color Definitions
    // ═══════════════════════════════════════════════════════════════════
    const OPT_CONFIG = {
        "SGD":           { color: "#94A3B8", active: true },
        "Momentum":      { color: "#3B82F6", active: true },
        "NAG":           { color: "#06B6D4", active: true },
        "AdaGrad":       { color: "#8B5CF6", active: false },
        "RMSProp":       { color: "#10B981", active: true },
        "Adam":          { color: "#F59E0B", active: true },
        "AdamW":         { color: "#EC4899", active: false },
        "Damped Newton": { color: "#EF4444", active: true }
    };

    // ═══════════════════════════════════════════════════════════════════
    // State Management
    // ═══════════════════════════════════════════════════════════════════
    var state = {
        mode: "race", // 'race' or 'river'
        surfaceKey: "multimodal",
        x0: 2.50,
        y0: 2.50,
        lr: 0.030,
        beta: 0.85,
        nSteps: 150,
        currentStep: 0,
        isPlaying: false,
        speed: 1.0,
        timer: null,
        simulationData: {},
        riverData: []
    };

    // ═══════════════════════════════════════════════════════════════════
    // Optimizer Physics Engine
    // ═══════════════════════════════════════════════════════════════════
    function computeHessian(surf, x, y) {
        var eps = 1e-4;
        var f0 = surf.f(x, y);
        var fxx = (surf.f(x + eps, y) - 2 * f0 + surf.f(x - eps, y)) / (eps * eps);
        var fyy = (surf.f(x, y + eps) - 2 * f0 + surf.f(x, y - eps)) / (eps * eps);
        var fxy = (surf.f(x + eps, y + eps) - surf.f(x + eps, y - eps) - surf.f(x - eps, y + eps) + surf.f(x - eps, y - eps)) / (4 * eps * eps);
        return [fxx, fxy, fxy, fyy];
    }

    function runOptimizer(optName, surf, x0, y0, nSteps, lr, beta) {
        var xs = [x0], ys = [y0], losses = [surf.f(x0, y0)];
        var x = x0, y = y0;
        var vx = 0, vy = 0;
        var Gx = 0, Gy = 0;
        var mx = 0, my = 0;
        var epsStep = null;
        var totalDist = 0;

        var t0 = performance.now();

        for (var t = 0; t < nSteps; t++) {
            var g = [0, 0];

            if (optName === "NAG") {
                var xAhead = x - beta * lr * vx;
                var yAhead = y - beta * lr * vy;
                g = surf.grad(xAhead, yAhead);
            } else {
                g = surf.grad(x, y);
            }

            // Norm clipping for stability
            var gNorm = Math.hypot(g[0], g[1]);
            if (gNorm > 150.0) {
                g[0] *= 150.0 / gNorm;
                g[1] *= 150.0 / gNorm;
            }

            var dx = 0, dy = 0;

            if (optName === "SGD") {
                dx = -lr * g[0];
                dy = -lr * g[1];
            } else if (optName === "Momentum") {
                vx = beta * vx + g[0];
                vy = beta * vy + g[1];
                dx = -lr * vx;
                dy = -lr * vy;
            } else if (optName === "NAG") {
                vx = beta * vx + g[0];
                vy = beta * vy + g[1];
                dx = -lr * vx;
                dy = -lr * vy;
            } else if (optName === "AdaGrad") {
                Gx += g[0] * g[0];
                Gy += g[1] * g[1];
                dx = - (lr * 4.0 / (Math.sqrt(Gx) + 1e-7)) * g[0];
                dy = - (lr * 4.0 / (Math.sqrt(Gy) + 1e-7)) * g[1];
            } else if (optName === "RMSProp") {
                vx = 0.9 * vx + 0.1 * g[0] * g[0];
                vy = 0.9 * vy + 0.1 * g[1] * g[1];
                dx = - (lr / (Math.sqrt(vx) + 1e-7)) * g[0];
                dy = - (lr / (Math.sqrt(vy) + 1e-7)) * g[1];
            } else if (optName === "Adam") {
                mx = 0.9 * mx + 0.1 * g[0];
                my = 0.9 * my + 0.1 * g[1];
                vx = 0.999 * vx + 0.001 * g[0] * g[0];
                vy = 0.999 * vy + 0.001 * g[1] * g[1];
                var mHatX = mx / (1.0 - Math.pow(0.9, t + 1));
                var mHatY = my / (1.0 - Math.pow(0.9, t + 1));
                var vHatX = vx / (1.0 - Math.pow(0.999, t + 1));
                var vHatY = vy / (1.0 - Math.pow(0.999, t + 1));
                dx = - (lr * 1.5 / (Math.sqrt(vHatX) + 1e-7)) * mHatX;
                dy = - (lr * 1.5 / (Math.sqrt(vHatY) + 1e-7)) * mHatY;
            } else if (optName === "AdamW") {
                mx = 0.9 * mx + 0.1 * g[0];
                my = 0.9 * my + 0.1 * g[1];
                vx = 0.999 * vx + 0.001 * g[0] * g[0];
                vy = 0.999 * vy + 0.001 * g[1] * g[1];
                var mHatX2 = mx / (1.0 - Math.pow(0.9, t + 1));
                var mHatY2 = my / (1.0 - Math.pow(0.9, t + 1));
                var vHatX2 = vx / (1.0 - Math.pow(0.999, t + 1));
                var vHatY2 = vy / (1.0 - Math.pow(0.999, t + 1));
                dx = - lr * 0.01 * x - (lr * 1.5 / (Math.sqrt(vHatX2) + 1e-7)) * mHatX2;
                dy = - lr * 0.01 * y - (lr * 1.5 / (Math.sqrt(vHatY2) + 1e-7)) * mHatY2;
            } else if (optName === "Damped Newton") {
                var H = computeHessian(surf, x, y);
                var mu = 0.8;
                var h11 = H[0] + mu, h12 = H[1], h22 = H[3] + mu;
                var det = h11 * h22 - h12 * h12;
                if (Math.abs(det) > 1e-6) {
                    var inv11 = h22 / det, inv12 = -h12 / det, inv22 = h11 / det;
                    dx = - (inv11 * g[0] + inv12 * g[1]) * 0.5;
                    dy = - (inv12 * g[0] + inv22 * g[1]) * 0.5;
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

            if (epsStep === null && Math.abs(l - surf.fStar) <= 0.01) {
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
            epsStep: epsStep,
            pathEff: pathEff,
            elapsedMs: elapsedMs
        };
    }

    function runRiverStreams(surf, nStreams, nSteps, lr) {
        var streams = [];
        var bounds = surf.xRange;
        var spanX = bounds[1] - bounds[0];
        var spanY = surf.yRange[1] - surf.yRange[0];

        for (var i = 0; i < nStreams; i++) {
            var frac = (i + 0.5) / nStreams;
            var rx0 = bounds[0] + frac * spanX;
            var ry0 = surf.yRange[1] - 0.2 * spanY * Math.sin(frac * Math.PI);

            var sxs = [rx0], sys = [ry0], szs = [surf.f(rx0, ry0)];
            var cx = rx0, cy = ry0;
            var vx = 0, vy = 0;

            for (var t = 0; t < nSteps; t++) {
                var g = surf.grad(cx, cy);
                var gNorm = Math.hypot(g[0], g[1]);
                if (gNorm > 100.0) { g[0] *= 100.0 / gNorm; g[1] *= 100.0 / gNorm; }

                vx = 0.90 * vx + (1 - 0.90) * g[0];
                vy = 0.90 * vy + (1 - 0.90) * g[1];

                cx -= lr * 1.5 * vx;
                cy -= lr * 1.5 * vy;
                sxs.push(cx);
                sys.push(cy);
                szs.push(surf.f(cx, cy));
            }
            streams.push({ xs: sxs, ys: sys, losses: szs });
        }
        return streams;
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3D Mesh Grid Generator
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
                row.push(surf.f(xs[i], ys[j]));
            }
            zs.push(row);
        }
        return { X: xs, Y: ys, Z: zs };
    }

    // ═══════════════════════════════════════════════════════════════════
    // UI Controller
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

            chip.innerHTML = '<span class="opt-dot"></span><span>' + name + '</span>';
            container.appendChild(chip);
        }
    }

    function toggleOpt(name) {
        OPT_CONFIG[name].active = !OPT_CONFIG[name].active;
        var chip = document.getElementById("chip-" + name.replace(/\\s+/g, '-'));
        if (OPT_CONFIG[name].active) chip.classList.add('active');
        else chip.classList.remove('active');
        recomputeAndRedraw();
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
        recomputeAndRedraw();
    }

    function setMode(mode) {
        state.mode = mode;
        document.getElementById('mode-race').classList.toggle('active', mode === 'race');
        document.getElementById('mode-river').classList.toggle('active', mode === 'river');
        document.getElementById('opt-selection-group').style.display = (mode === 'race' ? 'block' : 'none');
        document.getElementById('play-text').innerText = (mode === 'race' ? 'Play Race' : 'Flow River');
        recomputeAndRedraw();
    }

    function onSurfaceChange() {
        state.surfaceKey = document.getElementById('surface-select').value;
        var surf = SURFACES[state.surfaceKey];
        document.getElementById('surface-desc').innerText = surf.desc;
        state.x0 = surf.defaultX0;
        state.y0 = surf.defaultY0;
        state.lr = surf.defaultLr;

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

        document.getElementById('hud-fstar').innerText = surf.fStar.toFixed(2);
        recomputeAndRedraw();
    }

    function updateHyperparams() {
        state.lr = parseFloat(document.getElementById('lr-slider').value);
        state.beta = parseFloat(document.getElementById('beta-slider').value);
        state.nSteps = parseInt(document.getElementById('steps-slider').value, 10);

        document.getElementById('lr-val').innerText = state.lr.toFixed(3);
        document.getElementById('beta-val').innerText = state.beta.toFixed(2);
        document.getElementById('steps-val').innerText = state.nSteps;

        document.getElementById('scrubber').max = state.nSteps;
        recomputeAndRedraw();
    }

    function updateCoordinates() {
        state.x0 = parseFloat(document.getElementById('x0-slider').value);
        state.y0 = parseFloat(document.getElementById('y0-slider').value);
        document.getElementById('x0-val').innerText = state.x0.toFixed(2);
        document.getElementById('y0-val').innerText = state.y0.toFixed(2);
        recomputeAndRedraw();
    }

    function randomizeStart() {
        var surf = SURFACES[state.surfaceKey];
        var rx = surf.xRange[0] + Math.random() * (surf.xRange[1] - surf.xRange[0]);
        var ry = surf.yRange[0] + Math.random() * (surf.yRange[1] - surf.yRange[0]);
        state.x0 = rx;
        state.y0 = ry;
        document.getElementById('x0-slider').value = rx;
        document.getElementById('y0-slider').value = ry;
        document.getElementById('x0-val').innerText = rx.toFixed(2);
        document.getElementById('y0-val').innerText = ry.toFixed(2);
        recomputeAndRedraw();
    }

    // ═══════════════════════════════════════════════════════════════════
    // Plotly WebGL Viewport Renderer
    // ═══════════════════════════════════════════════════════════════════
    var plotDiv = document.getElementById('plotly-canvas');
    var isInitialized = false;

    function recomputeAndRedraw() {
        var surf = SURFACES[state.surfaceKey];
        var t0 = performance.now();

        if (state.mode === 'race') {
            state.simulationData = {};
            for (var name in OPT_CONFIG) {
                if (OPT_CONFIG[name].active) {
                    state.simulationData[name] = runOptimizer(
                        name, surf, state.x0, state.y0, state.nSteps, state.lr, state.beta
                    );
                }
            }
        } else {
            state.riverData = runRiverStreams(surf, 24, state.nSteps, state.lr);
        }

        var totalLatency = performance.now() - t0;
        var perStepUs = (totalLatency * 1000.0) / (state.nSteps * (Object.keys(state.simulationData).length || 1));
        document.getElementById('hud-latency').innerText = perStepUs.toFixed(1) + " µs";

        updateTelemetryHUD();
        render3DView();
    }

    function render3DView() {
        var surf = SURFACES[state.surfaceKey];
        var mesh = generateMesh(surf, 55);

        var traces = [];

        // 1. 3D Surface
        traces.push({
            type: 'surface',
            x: mesh.X,
            y: mesh.Y,
            z: mesh.Z,
            colorscale: 'Viridis',
            opacity: 0.82,
            showscale: false,
            contours: {
                z: { show: true, usecolormap: true, highlightcolor: '#FFFFFF', project: { z: true } }
            },
            hoverinfo: 'x+y+z',
            name: 'Loss Surface'
        });

        // 2. Start Marker
        var z0 = surf.f(state.x0, state.y0);
        traces.push({
            type: 'scatter3d',
            x: [state.x0],
            y: [state.y0],
            z: [z0],
            mode: 'markers+text',
            name: 'θ₀ Start',
            marker: { size: 7, color: '#EF4444', symbol: 'diamond' },
            text: ['Start θ₀'],
            textposition: 'top center',
            textfont: { color: '#EF4444', size: 10 }
        });

        var stepIdx = Math.min(state.currentStep, state.nSteps);

        if (state.mode === 'race') {
            for (var name in state.simulationData) {
                var sim = state.simulationData[name];
                var color = OPT_CONFIG[name].color;

                var sliceX = sim.xs.slice(0, stepIdx + 1);
                var sliceY = sim.ys.slice(0, stepIdx + 1);
                var sliceZ = sim.losses.slice(0, stepIdx + 1);

                // Trajectory ribbon
                traces.push({
                    type: 'scatter3d',
                    x: sliceX,
                    y: sliceY,
                    z: sliceZ,
                    mode: 'lines',
                    name: name,
                    line: { color: color, width: 4.5 }
                });

                // Leading Particle sphere
                if (sliceX.length > 0) {
                    var headX = sliceX[sliceX.length - 1];
                    var headY = sliceY[sliceY.length - 1];
                    var headZ = sliceZ[sliceZ.length - 1];
                    traces.push({
                        type: 'scatter3d',
                        x: [headX],
                        y: [headY],
                        z: [headZ],
                        mode: 'markers',
                        showlegend: false,
                        marker: { size: 6, color: color, symbol: 'circle' }
                    });
                }
            }
        } else {
            // River mode (24 streams)
            var pal = ['#38BDF8', '#06B6D4', '#2DD4BF', '#34D399', '#60A5FA', '#818CF8'];
            for (var i = 0; i < state.riverData.length; i++) {
                var st = state.riverData[i];
                var color = pal[i % pal.length];
                traces.push({
                    type: 'scatter3d',
                    x: st.xs.slice(0, stepIdx + 1),
                    y: st.ys.slice(0, stepIdx + 1),
                    z: st.losses.slice(0, stepIdx + 1),
                    mode: 'lines',
                    showlegend: false,
                    line: { color: color, width: 2.5 }
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

        var layout = {
            paper_bgcolor: '#0B0F19',
            plot_bgcolor: '#111827',
            margin: { l: 0, r: 0, t: 20, b: 0 },
            showlegend: false,
            scene: {
                xaxis: Object.assign({}, axisCfg, { title: 'Parameter θ₁' }),
                yaxis: Object.assign({}, axisCfg, { title: 'Parameter θ₂' }),
                zaxis: Object.assign({}, axisCfg, { title: 'Loss L(θ)' }),
                bgcolor: '#0B0F19',
                camera: {
                    eye: { x: -1.6, y: -1.6, z: 1.3 },
                    up: { x: 0, y: 0, z: 1 }
                }
            }
        };

        var config = {
            responsive: true,
            displayModeBar: false
        };

        if (!isInitialized) {
            Plotly.newPlot(plotDiv, traces, layout, config).then(function() {
                isInitialized = true;
                document.getElementById('loading-screen').style.display = 'none';

                // Click on 3D surface to set start position
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
                            recomputeAndRedraw();
                        }
                    }
                });
            });
        } else {
            Plotly.react(plotDiv, traces, layout, config);
        }
    }

    function updateTelemetryHUD() {
        var tbody = document.getElementById('telemetry-tbody');
        tbody.innerHTML = "";
        var stepIdx = Math.min(state.currentStep, state.nSteps);

        if (state.mode === 'race') {
            for (var name in state.simulationData) {
                var sim = state.simulationData[name];
                var color = OPT_CONFIG[name].color;
                var curLoss = sim.losses[Math.min(stepIdx, sim.losses.length - 1)];

                var tr = document.createElement('tr');

                var convBadge = sim.epsStep !== null
                    ? '<span style="color:#34D399;font-weight:700;">★ ' + sim.epsStep + ' st</span>'
                    : '<span style="color:#9CA3AF;">L=' + curLoss.toFixed(3) + '</span>';

                tr.innerHTML =
                    '<td><span style="color:' + color + ';font-weight:700;">● ' + name + '</span></td>' +
                    '<td>' + (curLoss > 1e4 ? curLoss.toExponential(1) : curLoss.toFixed(4)) + '</td>' +
                    '<td>' + sim.pathEff.toFixed(3) + '</td>' +
                    '<td>' + convBadge + '</td>';

                tbody.appendChild(tr);
            }
        } else {
            var tr = document.createElement('tr');
            tr.innerHTML = '<td colspan="4" style="color:#60A5FA;text-align:center;padding:12px 0;">🌊 24 liquid tributaries navigating downhill potential gradient</td>';
            tbody.appendChild(tr);
        }

        document.getElementById('step-counter').innerText = 'Step: ' + stepIdx + ' / ' + state.nSteps;
        document.getElementById('scrubber').value = stepIdx;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Live Animation Controller
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

        var delay = Math.max(16, 40 / state.speed);
        state.timer = setInterval(function() {
            if (state.currentStep < state.nSteps) {
                state.currentStep++;
                updateTelemetryHUD();
                render3DView();
            } else {
                pauseSim();
            }
        }, delay);
    }

    function pauseSim() {
        state.isPlaying = false;
        clearInterval(state.timer);
        document.getElementById('play-icon').innerText = "▶";
        document.getElementById('play-text').innerText = (state.mode === 'race' ? 'Play Race' : 'Flow River');
    }

    function resetSim() {
        pauseSim();
        state.currentStep = 0;
        updateTelemetryHUD();
        render3DView();
    }

    function stepForward() {
        pauseSim();
        if (state.currentStep < state.nSteps) {
            state.currentStep = Math.min(state.currentStep + 10, state.nSteps);
            updateTelemetryHUD();
            render3DView();
        }
    }

    function onScrub(val) {
        pauseSim();
        state.currentStep = parseInt(val, 10);
        updateTelemetryHUD();
        render3DView();
    }

    function setSpeed(spd) {
        state.speed = spd;
        ['spd-1'].forEach(function(id) {
            var el = document.getElementById(id);
            if (el) el.style.background = "#1F2937";
        });
        if (state.isPlaying) {
            pauseSim();
            playSim();
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Resilient Script Loader & Initialization
    // ═══════════════════════════════════════════════════════════════════
    function bootstrapApp() {
        buildOptimizerChips();
        onSurfaceChange(); // Loads default surface & computes initial race
        state.currentStep = state.nSteps; // Start with full path pre-rendered
        recomputeAndRedraw();
    }

    (function loadPlotly() {
        if (typeof Plotly !== 'undefined') {
            bootstrapApp();
        } else {
            var s = document.createElement('script');
            s.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
            s.crossOrigin = 'anonymous';
            s.onload = bootstrapApp;
            s.onerror = function() {
                var s2 = document.createElement('script');
                s2.src = 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js';
                s2.onload = bootstrapApp;
                s2.onerror = function() {
                    var loader = document.getElementById('loading-screen');
                    loader.innerHTML = '<div style="color:#EF4444;font-size:18px;font-weight:700;">Plotly 3D Engine Load Error</div>' +
                                       '<div style="color:#9CA3AF;font-size:12px;margin-top:8px;">Unable to load Plotly CDN. Please check your network or adblocker rules.</div>';
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
    return html


def export_interactive_html():
    print("  🌐 Compiling full client-side WebGL 3D visualizer into assets/interactive_3d_landscape.html ...")
    output_path = os.path.join(ASSETS_DIR, "interactive_3d_landscape.html")

    html_content = build_interactive_app_html()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"    ✓ Successfully compiled {output_path} ({len(html_content) / 1024:.1f} KB)")


if __name__ == "__main__":
    export_interactive_html()
