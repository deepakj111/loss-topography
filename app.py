"""
Interactive Streamlit + Plotly Web Laboratory for loss-topography.

Run with: streamlit run app.py
"""

from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from loss_topography.theme import (
    OPTIMIZER_COLORS, plotly_layout_defaults, plotly_3d_scene,
    BG_CANVAS, BG_PANEL, GRID_COLOR, TEXT_HEADER, TEXT_LABEL, TEXT_TICK,
)
from loss_topography.surfaces import SURFACE_REGISTRY, ALL_SURFACES
from loss_topography.optimizers import OPTIMIZER_CLASSES, get_default_optimizers
from loss_topography.losses import ALL_LOSSES, LOSS_REGISTRY
from loss_topography.sharpness import smooth_twin_well_1d, compute_sharpness

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Loss Topography — Interactive ML Lab",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG_CANVAS};
    }}
    .stSidebar {{
        background-color: {BG_PANEL};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {BG_PANEL};
        color: {TEXT_HEADER};
        border-radius: 4px;
        padding: 8px 16px;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background-color: #1E3A5F;
    }}
    h1, h2, h3 {{
        color: {TEXT_HEADER} !important;
    }}
    .metric-card {{
        background: {BG_PANEL};
        border: 1px solid {GRID_COLOR};
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
    }}
</style>
""", unsafe_allow_html=True)

st.title("🏔️ Loss Topography — Interactive ML Laboratory")
st.caption("Explore loss landscapes, critical points, optimizer physics, and neural network surfaces")


# ── Tab layout ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏁 3D Optimizer Race",
    "🔍 Saddle & Curvature Inspector",
    "📊 Sharp vs Flat Minima",
    "📈 Loss Function Sandbox",
    "🧠 NN Landscape Slicer",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1: 3D Optimizer Race Arena
# ═══════════════════════════════════════════════════════════════════

with tab1:
    col_ctrl, col_viz = st.columns([1, 3])

    with col_ctrl:
        st.subheader("⚙️ Controls")
        surface_name = st.selectbox(
            "Surface", list(SURFACE_REGISTRY.keys()),
            index=list(SURFACE_REGISTRY.keys()).index("Multi-Modal Arena"),
            key="race_surface",
        )
        surface = SURFACE_REGISTRY[surface_name]

        x0 = st.slider("Start x₀", float(surface.x_range[0]),
                        float(surface.x_range[1]), 2.5, 0.1, key="race_x0")
        y0 = st.slider("Start y₀", float(surface.y_range[0]),
                        float(surface.y_range[1]), 2.5, 0.1, key="race_y0")
        lr = st.slider("Learning Rate", 0.001, 0.2, 0.03, 0.001, key="race_lr")
        n_steps = st.slider("Steps", 50, 500, 200, 10, key="race_steps")

        selected_opts = st.multiselect(
            "Optimizers",
            list(OPTIMIZER_COLORS.keys()),
            default=["SGD", "Momentum", "Adam", "RMSProp"],
            key="race_opts",
        )

    with col_viz:
        if st.button("🚀 Run Race!", key="run_race"):
            # Compute surface
            X, Y, Z = surface.meshgrid(n=100)
            Z_clip = np.clip(Z, np.percentile(Z, 2), np.percentile(Z, 98))

            # 3D surface
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "surface"}, {"type": "xy"}]],
                subplot_titles=["3D Loss Surface", "2D Contour + Trajectories"],
            )

            fig.add_trace(
                go.Surface(x=X, y=Y, z=Z_clip, colorscale="Viridis",
                           opacity=0.7, showscale=False),
                row=1, col=1,
            )

            fig.add_trace(
                go.Contour(x=X[0], y=Y[:, 0], z=Z_clip,
                           colorscale="Viridis", showscale=False,
                           contours=dict(coloring="heatmap"),
                           opacity=0.6),
                row=1, col=2,
            )

            # Run optimizers
            for opt_name in selected_opts:
                cls = OPTIMIZER_CLASSES.get(opt_name)
                if cls is None:
                    continue
                if opt_name in ("AdaGrad",):
                    opt = cls(lr=lr * 10)
                elif opt_name == "Damped Newton":
                    opt = cls(lr=lr * 20, mu=1.0)
                else:
                    opt = cls(lr=lr) if opt_name == "SGD" else cls(lr=lr)

                log = opt.run(surface, x0, y0, n_steps=n_steps)
                color = OPTIMIZER_COLORS[opt_name]

                # 3D trajectory
                fig.add_trace(
                    go.Scatter3d(
                        x=log.xs, y=log.ys, z=log.losses,
                        mode="lines+markers",
                        marker=dict(size=2, color=color),
                        line=dict(color=color, width=3),
                        name=opt_name,
                    ),
                    row=1, col=1,
                )

                # 2D trajectory
                fig.add_trace(
                    go.Scatter(
                        x=log.xs, y=log.ys,
                        mode="lines+markers",
                        marker=dict(size=3, color=color),
                        line=dict(color=color, width=2),
                        name=f"{opt_name} (2D)",
                        showlegend=False,
                    ),
                    row=1, col=2,
                )

            layout = plotly_layout_defaults()
            scene_cfg = plotly_3d_scene()
            fig.update_layout(
                **layout,
                height=600,
                **scene_cfg,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrics
            st.subheader("📊 Race Results")
            cols = st.columns(len(selected_opts))
            for i, opt_name in enumerate(selected_opts):
                cls = OPTIMIZER_CLASSES.get(opt_name)
                if cls is None:
                    continue
                if opt_name in ("AdaGrad",):
                    opt = cls(lr=lr * 10)
                elif opt_name == "Damped Newton":
                    opt = cls(lr=lr * 20, mu=1.0)
                else:
                    opt = cls(lr=lr)
                log = opt.run(surface, x0, y0, n_steps=n_steps)
                with cols[i]:
                    st.markdown(f"""
                    <div class='metric-card'>
                    <b style='color:{OPTIMIZER_COLORS[opt_name]}'>{opt_name}</b><br>
                    Final Loss: {log.losses[-1]:.4f}<br>
                    Path Eff: {log.path_efficiency:.3f}<br>
                    Converged: {log.converged_step or '—'}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👆 Configure settings and click **Run Race!**")


# ═══════════════════════════════════════════════════════════════════
# TAB 2: Saddle Point & Curvature Inspector
# ═══════════════════════════════════════════════════════════════════

with tab2:
    col_ctrl2, col_viz2 = st.columns([1, 3])

    with col_ctrl2:
        st.subheader("🔍 Inspect Point")
        surface_name2 = st.selectbox(
            "Surface", list(SURFACE_REGISTRY.keys()),
            index=list(SURFACE_REGISTRY.keys()).index("Regularized Saddle"),
            key="inspect_surface",
        )
        surface2 = SURFACE_REGISTRY[surface_name2]

        px_val = st.slider("x", float(surface2.x_range[0]),
                           float(surface2.x_range[1]), 0.0, 0.05,
                           key="inspect_x")
        py_val = st.slider("y", float(surface2.y_range[0]),
                           float(surface2.y_range[1]), 0.0, 0.05,
                           key="inspect_y")

    with col_viz2:
        X2, Y2, Z2 = surface2.meshgrid(n=100)
        Z2_clip = np.clip(Z2, np.percentile(Z2, 2), np.percentile(Z2, 98))

        fig2 = go.Figure()
        fig2.add_trace(
            go.Contour(x=X2[0], y=Y2[:, 0], z=Z2_clip,
                       colorscale="Viridis", showscale=True,
                       contours=dict(coloring="heatmap"))
        )

        # Compute Hessian at selected point
        try:
            H = surface2.hessian(px_val, py_val)
            eigvals, eigvecs = np.linalg.eigh(H)

            # Draw eigenvectors
            scale = 0.5
            for k in range(2):
                ev = eigvecs[:, k] * scale
                color = "#10B981" if eigvals[k] > 0 else "#EF4444"
                fig2.add_trace(go.Scatter(
                    x=[px_val, px_val + ev[0]],
                    y=[py_val, py_val + ev[1]],
                    mode="lines+markers",
                    line=dict(color=color, width=3),
                    marker=dict(size=[8, 12], color=color,
                                symbol=["circle", "arrow-up"]),
                    name=f"λ{k+1}={eigvals[k]:.3f}",
                ))

            fig2.add_trace(go.Scatter(
                x=[px_val], y=[py_val],
                mode="markers",
                marker=dict(size=12, color="#FBBF24", symbol="x"),
                name="Inspection Point",
            ))
        except Exception:
            pass

        layout2 = plotly_layout_defaults()
        fig2.update_layout(
            **layout2,
            height=550,
            title=f"Curvature at ({px_val:.2f}, {py_val:.2f})",
            xaxis_title="θ₁", yaxis_title="θ₂",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Show Hessian info
        try:
            H = surface2.hessian(px_val, py_val)
            eigvals, eigvecs = np.linalg.eigh(H)
            f_val = surface2.f_scalar(px_val, py_val)

            c1, c2, c3 = st.columns(3)
            c1.metric("f(x,y)", f"{f_val:.4f}")
            c2.metric("λ₁", f"{eigvals[0]:.4f}")
            c3.metric("λ₂", f"{eigvals[1]:.4f}")

            if eigvals[0] * eigvals[1] < 0:
                st.warning("⚠️ **Saddle Point Region** — Mixed eigenvalue signs!")
            elif eigvals[0] > 0 and eigvals[1] > 0:
                st.success("✅ **Convex Region** — Both eigenvalues positive")
            elif eigvals[0] == 0 or eigvals[1] == 0:
                st.info("ℹ️ **Degenerate** — At least one zero eigenvalue")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# TAB 3: Sharp vs Flat Generalization Lab
# ═══════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Sharp vs. Flat Minima — Why Flat Minima Generalize Better")

    col_ctrl3, col_viz3 = st.columns([1, 3])

    with col_ctrl3:
        shift = st.slider("Distribution Shift Δθ", 0.0, 1.0, 0.3, 0.01,
                           key="shift_val")
        sharp_curv = st.slider("Sharp Curvature", 5.0, 100.0, 40.0, 1.0,
                               key="sharp_curv")
        flat_curv = st.slider("Flat Curvature", 0.1, 5.0, 1.0, 0.1,
                              key="flat_curv")

    with col_viz3:
        theta = np.linspace(-5, 5, 500)
        train_loss = smooth_twin_well_1d(theta, sharp_curv, flat_curv)
        test_loss = smooth_twin_well_1d(theta - shift, sharp_curv, flat_curv)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=theta, y=train_loss, mode="lines",
            line=dict(color="#06B6D4", width=3),
            name="Train Loss",
        ))
        fig3.add_trace(go.Scatter(
            x=theta, y=test_loss, mode="lines",
            line=dict(color="#EC4899", width=2, dash="dash"),
            name="Test Loss (shifted)",
        ))

        # Mark minima
        fig3.add_vline(x=-2, line=dict(color="#EF4444", dash="dot", width=1))
        fig3.add_vline(x=2, line=dict(color="#10B981", dash="dot", width=1))

        fig3.add_annotation(x=-2, y=0.2, text="Sharp Min",
                            font=dict(color="#EF4444", size=12),
                            showarrow=False)
        fig3.add_annotation(x=2, y=0.2, text="Flat Min",
                            font=dict(color="#10B981", size=12),
                            showarrow=False)

        layout3 = plotly_layout_defaults()
        fig3.update_layout(
            **layout3,
            height=450,
            xaxis_title="θ (parameter)",
            yaxis_title="Loss",
            title=f"Distribution Shift: Δθ = {shift:.2f}",
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Metrics
        sharp_train = float(smooth_twin_well_1d(np.array([-2]), sharp_curv, flat_curv)[0])
        sharp_test = float(smooth_twin_well_1d(np.array([-2 + shift]), sharp_curv, flat_curv)[0])
        flat_train = float(smooth_twin_well_1d(np.array([2]), sharp_curv, flat_curv)[0])
        flat_test = float(smooth_twin_well_1d(np.array([2 + shift]), sharp_curv, flat_curv)[0])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("🔴 Sharp: Test Loss",
                      f"{sharp_test:.4f}",
                      delta=f"{sharp_test - sharp_train:+.4f}")
        with c2:
            st.metric("🟢 Flat: Test Loss",
                      f"{flat_test:.4f}",
                      delta=f"{flat_test - flat_train:+.4f}")


# ═══════════════════════════════════════════════════════════════════
# TAB 4: Loss Function Sandbox
# ═══════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Loss Function Outlier & Convergence Sandbox")

    col_ctrl4, col_viz4 = st.columns([1, 3])

    with col_ctrl4:
        selected_losses = st.multiselect(
            "Losses to compare",
            [l.name for l in ALL_LOSSES],
            default=["MSE", "MAE", "Huber", "Cauchy"],
            key="sandbox_losses",
        )

        n_outliers = st.slider("Number of Outliers", 0, 10, 3,
                               key="n_outliers")
        outlier_scale = st.slider("Outlier Magnitude", 1.0, 20.0, 8.0, 0.5,
                                  key="outlier_scale")

    with col_viz4:
        r = np.linspace(-5, 5, 300)

        fig4 = make_subplots(rows=1, cols=2,
                             subplot_titles=["Loss Value L(r)", "Gradient ∇L(r)"])

        loss_colors = ["#06B6D4", "#F59E0B", "#10B981", "#A855F7",
                       "#EC4899", "#3B82F6", "#EF4444", "#FBBF24",
                       "#8B5CF6", "#F43F5E"]

        for i, lname in enumerate(selected_losses):
            loss = LOSS_REGISTRY.get(lname)
            if loss is None:
                continue
            color = loss_colors[i % len(loss_colors)]
            vals = np.clip(loss.value(r), -10, 10)
            grads = np.clip(loss.grad(r), -10, 10)

            fig4.add_trace(
                go.Scatter(x=r, y=vals, mode="lines",
                           line=dict(color=color, width=2),
                           name=lname),
                row=1, col=1,
            )
            fig4.add_trace(
                go.Scatter(x=r, y=grads, mode="lines",
                           line=dict(color=color, width=2, dash="dash"),
                           name=f"{lname} ∇", showlegend=False),
                row=1, col=2,
            )

        layout4 = plotly_layout_defaults()
        fig4.update_layout(**layout4, height=450)
        st.plotly_chart(fig4, use_container_width=True)

        # Outlier regression demo
        if n_outliers > 0:
            st.markdown("#### Outlier Robustness Demo")
            rng = np.random.RandomState(42)
            x_data = np.linspace(0, 5, 30)
            y_clean = 2 * x_data + 1 + rng.randn(30) * 0.5

            # Add outliers
            outlier_idx = rng.choice(30, n_outliers, replace=False)
            y_data = y_clean.copy()
            y_data[outlier_idx] += outlier_scale * rng.choice([-1, 1], n_outliers)

            fig_out = go.Figure()
            fig_out.add_trace(go.Scatter(
                x=x_data, y=y_data, mode="markers",
                marker=dict(size=8, color="#06B6D4"),
                name="Data",
            ))

            # Highlight outliers
            fig_out.add_trace(go.Scatter(
                x=x_data[outlier_idx], y=y_data[outlier_idx],
                mode="markers",
                marker=dict(size=12, color="#EF4444", symbol="x"),
                name="Outliers",
            ))

            fig_out.update_layout(**plotly_layout_defaults(), height=350,
                                  title="Regression with Outliers",
                                  xaxis_title="x", yaxis_title="y")
            st.plotly_chart(fig_out, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 5: Real PyTorch NN Landscape Slicer
# ═══════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("🧠 Filter-Normalized Neural Network Loss Landscape")
    st.markdown("""
    Train a live mini-MLP and explore its 2D filter-normalized loss slice
    *(Li et al., 2018)*. Compare **plain** vs. **residual (skip connection)** networks.
    """)

    col_ctrl5, col_viz5 = st.columns([1, 3])

    with col_ctrl5:
        use_skip = st.checkbox("Use Skip Connections", value=True, key="nn_skip")
        grid_sz = st.slider("Grid Size", 11, 31, 21, 2, key="nn_grid")
        n_epochs = st.slider("Training Epochs", 50, 500, 150, 10, key="nn_epochs")
        hidden_sz = st.slider("Hidden Size", 16, 128, 32, 8, key="nn_hidden")

    with col_viz5:
        if st.button("🔬 Train & Visualize", key="nn_train"):
            with st.spinner("Training neural network and computing landscape..."):
                try:
                    from loss_topography.nn_landscape import train_and_visualize_landscape
                    result = train_and_visualize_landscape(
                        use_skip=use_skip, grid_size=grid_sz,
                        alpha_range=1.0, n_epochs=n_epochs,
                        hidden=hidden_sz, seed=42,
                    )

                    A = result["alpha_grid"]
                    B = result["beta_grid"]
                    Z = result["loss_surface"]
                    Z_clip = np.clip(Z, np.percentile(Z, 1), np.percentile(Z, 99))

                    fig5 = make_subplots(
                        rows=1, cols=2,
                        specs=[[{"type": "surface"}, {"type": "xy"}]],
                        subplot_titles=["3D Loss Landscape", "2D Contour"],
                    )

                    fig5.add_trace(
                        go.Surface(x=A, y=B, z=Z_clip, colorscale="Viridis",
                                   opacity=0.8, showscale=False),
                        row=1, col=1,
                    )

                    fig5.add_trace(
                        go.Contour(x=A[0], y=B[:, 0], z=Z_clip,
                                   colorscale="Viridis", showscale=True),
                        row=1, col=2,
                    )

                    layout5 = plotly_layout_defaults()
                    scene5 = plotly_3d_scene()
                    fig5.update_layout(**layout5, **scene5, height=500,
                                       title=result.get("model_info", ""))
                    st.plotly_chart(fig5, use_container_width=True)

                    st.success(f"✅ {result.get('model_info', 'Done')}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("👆 Configure and click **Train & Visualize**")
