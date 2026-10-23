<div align="center">

# 🏔️ loss-topography

### An Interactive & Animated Visual Laboratory for Machine Learning Loss Surfaces, Optimization Dynamics, and Landscape Geometry

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly 3D](https://img.shields.io/badge/Plotly_3D-Interactive-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Tests Passing](https://img.shields.io/badge/tests-67%20passed-brightgreen.svg?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b>Explore why modern neural networks train, why optimizers fail or succeed, and how loss geometry governs generalization.</b>
  <br />
  Featuring 10 loss functions, 9 mathematical landscapes, 8 optimizers, Hessian eigenspectrum diagnostics, distribution shift simulations, and filter-normalized neural network slices.
  <br />
  <br />
  <a href="#-quickstart--local-replication"><b>⚡ Replicate Locally (10s)</b></a> &bull;
  <a href="#-interactive-web-laboratory-streamlit--plotly"><b>🖥️ Launch Streamlit Lab</b></a> &bull;
  <a href="#-zero-install-interactive-3d-visualizer"><b>🌐 Open 3D HTML Viewer</b></a> &bull;
  <a href="#-theoretical-foundations"><b>📐 Mathematical Foundations</b></a>
</p>

---

</div>

## 🌊 The River of Optimization

> *"Optimization dynamics mirror fluid mechanics: gradient descent acts like water flowing through mountain passes, seeking gravitational potential minimum."*

Watch an ensemble of 28 optimization streams flow downhill across rugged non-convex multi-modal topography. The streams begin along the high-elevation mountain ridge, branch through natural passes, dynamically navigate saddle barriers, and pool like bioluminescent liquid tributaries into the global minimum basin:

<div align="center">
  <img src="assets/optimization_river_flow.gif" alt="Optimization River Flow" width="100%" />
</div>

---

## 🌐 Zero-Install Interactive 3D Visualizer

You don't even need to install Python to explore the 3D loss surface and trajectories interactively!

We have compiled a standalone, self-contained **WebGL + Plotly 3D laboratory** directly into the repository:
- 🔗 **[`assets/interactive_3d_landscape.html`](assets/interactive_3d_landscape.html)**

**How to open:**
- Double-click `assets/interactive_3d_landscape.html` in your file explorer to open it in Chrome, Firefox, Safari, or Edge.
- **Controls:** Click & drag to rotate 360° &bull; Scroll to zoom &bull; Double-click to reset camera &bull; Hover over trajectories for $(x, y, \mathcal{L})$ telemetry &bull; Click legend items to isolate individual optimizers.

---

## 🌟 Visual Showcase & Animations

### 1. The Multi-Modal Optimizer Race
Watch **SGD**, **Momentum**, **RMSProp**, and **Adam** compete in real time across rugged, non-convex terrain with multiple local basins and saddle barriers. Notice how adaptive learning rates (RMSProp, Adam) navigate ravines while standard SGD stalls or oscillates violently across steep walls.

<div align="center">
  <img src="assets/hero_optimizer_race.gif" alt="3D Optimizer Race" width="100%" />
</div>

---

### 2. The Comprehensive Loss Function Atlas
A side-by-side taxonomy of 10 classic and modern loss functions with their exact analytical derivatives $\nabla \mathcal{L}(r)$. Observe the smooth saturation of Huber and Cauchy vs. the discontinuity in subgradient MAE, and the asymmetric probability-shaping of Focal Loss.

<div align="center">
  <img src="assets/loss_function_atlas.png" alt="Loss Function Atlas" width="100%" />
</div>

---

### 3. Saddle Point Escape & Negative Curvature
Non-convex optimization in high dimensions is dominated by saddle points, not local minima. Here, SGD without momentum remains trapped in the low-gradient plateau of a strict saddle ($\lambda_1 < 0, \lambda_2 > 0$), whereas Momentum and NAG build velocity along the negative curvature direction to escape rapidly.

<div align="center">
  <img src="assets/saddle_point_escape.gif" alt="Saddle Point Escape" width="100%" />
</div>

---

### 4. Sharp vs. Flat Minima & The Generalization Gap
Why do flat minima generalize better than sharp minima? (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017). Under dataset distribution shift $P_{\text{train}} \neq P_{\text{test}}$ ($\theta^* \to \theta^* + \Delta\theta$), training loss remains low for both, but the sharp basin suffers catastrophic test error explosion while the flat basin maintains robust generalization.

<div align="center">
  <img src="assets/sharp_vs_flat_minima.gif" alt="Sharp vs Flat Minima Generalization" width="100%" />
</div>

---

### 5. Empirical Convergence & Wall-Clock Efficiency
Is an optimizer fast per step, or fast per unit of progress? We evaluate optimizers along two orthogonal axes: **Iteration Complexity** (steps to reach $\epsilon$-ball) and **Wall-Clock Computational Overhead** ($\mu\text{s}$ per gradient update step), alongside **Trajectory Path Efficiency** (ratio of Euclidean displacement to total path length).

<div align="center">
  <img src="assets/loss_convergence_timing.png" alt="Convergence and Timing Benchmark" width="100%" />
</div>

---

### 6. Filter-Normalized Neural Network Loss Landscapes
Directly replicating the landmark technique from Li et al. (2018), *"Visualizing the Loss Landscape of Neural Nets"*. By applying filter-wise scale normalization $\delta_{l,j} = \frac{d_{l,j}}{\|d_{l,j}\|_F} \|\theta^*_{l,j}\|_F$, we eliminate scale-invariance artifacts and expose the dramatic geometric smoothing effect introduced by skip connections.

<div align="center">
  <img src="assets/filter_normalized_nn_surface.png" alt="Filter Normalized NN Surface" width="100%" />
</div>

---

## ⚡ Quickstart & Local Replication

Every visual artifact in this repository can be replicated locally in under 10 seconds.

### 1. One-Line Replication Runner
```bash
# Clone and install dependencies
git clone https://github.com/deepakj111/loss-topography.git
cd loss-topography
pip install -r requirements.txt

# Run the instant replication suite (generates plots in seconds)
python scripts/quickstart_replicate.py
```

### 2. Three-Line Python Snippets

#### Replicate an Optimizer Trajectory & Telemetry
```python
from loss_topography.surfaces import Rosenbrock
from loss_topography.optimizers import Adam, MomentumSGD

surface = Rosenbrock()
adam = Adam(lr=0.04)
log = adam.run(surface, x0=-0.8, y0=1.2, n_steps=100)

print(f"Final Loss:      {log.losses[-1]:.4f}")
print(f"Path Efficiency: {log.path_efficiency:.3f}")   # Geodesic ratio
print(f"Step Latency:    {log.wall_clock_ms_per_step * 1000:.1f} µs")
```

#### Replicate Analytical Loss Function & Gradient
```python
import numpy as np
from loss_topography.losses import Huber, Cauchy, FocalLoss

residual = np.array([-2.0, 0.0, 1.5])
print("Huber Value:", Huber(residual))
print("Huber Grad: ", Huber.grad(residual))
```

#### Replicate Critical Point & Hessian Eigenspectrum
```python
from loss_topography.surfaces import StrictSaddle

saddle = StrictSaddle()
print("Curvature Type:", saddle.classify_point(0.0, 0.0))
print("Eigenvalues:   ", saddle.eigenvalues(0.0, 0.0))
```

---

## 🔬 Theoretical Foundations

### 1. Loss Functions & Analytical Derivatives

Every loss function in `loss-topography` implements closed-form vector evaluations and exact analytical first and second derivatives:

| Loss Function | Mathematical Formulation | Primary Domain & Use Case | Robustness to Outliers |
| :--- | :--- | :--- | :--- |
| **MSE ($L_2$)** | `½ (y - ŷ)²` | Gaussian noise regression | Low (quadratic penalty $\propto r^2$) |
| **MAE ($L_1$)** | `\|y - ŷ\|` | Laplace noise, median regression | High (linear penalty $\propto \|r\|$) |
| **Huber Loss** | `½ r²` (if `\|r\| ≤ δ`), else `δ(\|r\| - ½δ)` | Robust regression, RL value targets | High (smooth transition at $\delta$) |
| **Log-Cosh** | `log(cosh(y - ŷ))` | Smooth $L_1$ approximation | High (everywhere twice differentiable) |
| **Quantile (Pinball)** | `max(τ·r, (τ - 1)·r)` | Uncertainty estimation, risk bounds | High (asymmetric quantile regression) |
| **Binary Cross-Entropy** | `-[y·log(ŷ) + (1-y)·log(1-ŷ)]` | Probabilistic binary classification | Standard cross-entropy |
| **Focal Loss** | `-α_t (1 - p_t)^γ log(p_t)` | Severe class imbalance, object detection | Down-weights easy examples ($\gamma > 0$) |
| **Hinge Loss** | `max(0, 1 - y·ŷ)` | Maximum-margin classification (SVM) | Sparse support vectors |
| **KL Divergence** | `∑ p·log(p / q)` | Knowledge distillation, VAEs | Asymmetric information divergence |
| **Cauchy Loss** | `(c²/2) log(1 + r²/c²)` | Computer vision robust estimation | Extreme (descending influence curve) |

<details>
<summary><b>📐 Click to Expand Full LaTeX Definitions & Derivatives</b></summary>

#### Huber Loss
$$
\mathcal{L}_\delta(r) = \begin{cases} \frac{1}{2}r^2 & \text{if } |r| \le \delta \\ \delta(|r| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}, \qquad \nabla \mathcal{L}_\delta(r) = \begin{cases} r & \text{if } |r| \le \delta \\ \delta \operatorname{sgn}(r) & \text{otherwise} \end{cases}
$$

#### Focal Loss (Lin et al., ICCV 2017)
$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t), \qquad p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{otherwise} \end{cases}
$$

#### Cauchy / Lorentzian Loss
$$
\rho(r) = \frac{c^2}{2} \log\left(1 + \frac{r^2}{c^2}\right), \qquad \psi(r) = \frac{r}{1 + (r/c)^2}
$$

#### Log-Cosh Loss
$$
\mathcal{L}(r) = \log(\cosh(r)), \qquad \nabla \mathcal{L}(r) = \tanh(r), \qquad \nabla^2 \mathcal{L}(r) = \operatorname{sech}^2(r)
$$

</details>

---

### 2. Benchmark Topographies & Critical Point Geometries

The library provides 9 analytical surfaces with closed-form gradients $\nabla f$ and Hessian matrices $\mathbf{H} = \nabla^2 f$:

1. **Rosenbrock Banana Valley**: $f(x, y) = (a - x)^2 + b(y - x^2)^2$. Ill-conditioned curved parabolic valley where gradient descent oscillates endlessly across steep walls.
2. **Rastrigin Surface**: $f(x, y) = 20 + x^2 - 10\cos(2\pi x) + y^2 - 10\cos(2\pi y)$. Highly non-convex field riddled with dense lattices of local minima.
3. **Ackley Function**: Nearly flat outer plateau with an exponential central funnel. Demands momentum or adaptive step-sizes to avoid stalling.
4. **Beale & Booth Functions**: Anisotropic quadratic-like surfaces with sharply unequal condition numbers $\kappa = \lambda_{\max} / \lambda_{\min} \gg 1$.
5. **Himmelblau Multimodal**: Four distinct global minima with identical zero loss, serving as a clean test of initialization basin sensitivity.
6. **Strict Saddle**: $f(x, y) = x^2 - y^2$. Eigenvalues $\lambda_1 = 2, \lambda_2 = -2$. Pure hyperbolic saddle showcasing negative curvature escape.
7. **Monkey Saddle**: $f(x, y) = x^3 - 3xy^2$. Degenerate higher-order saddle where $\mathbf{H}(0, 0) = \mathbf{0}$, rendering standard Newton steps undefined.
8. **Regularized Saddle**: $f(x, y) = x^2 - y^2 + 0.1x^4 + 0.1y^4$. Escape along $y$ leads to bounded, stable symmetric double minima at $(0, \pm\sqrt{5})$ with minimum value $f^* = -2.5000$.

---

### 3. Optimization Algorithms

`loss-topography` implements 8 fundamental and modern optimizers with explicit trajectory logging:

- **Vanilla Gradient Descent (SGD)**:
  $$\theta_{t+1} = \theta_t - \eta \mathbf{g}_t$$

- **Polyak Momentum**:
  $$v_{t+1} = \beta v_t + \mathbf{g}_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}$$

- **Nesterov Accelerated Gradient (NAG)**:
  $$\theta_{\text{ahead}} = \theta_t - \beta \eta v_t, \qquad v_{t+1} = \beta v_t + \nabla f(\theta_{\text{ahead}}), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}$$

- **AdaGrad**:
  $$G_{t+1} = G_t + \mathbf{g}_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_{t+1}} + \epsilon} \odot \mathbf{g}_t$$

- **RMSProp**:
  $$v_{t+1} = \beta v_t + (1 - \beta) \mathbf{g}_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_{t+1}} + \epsilon} \odot \mathbf{g}_t$$

- **Adam**:
  $$m_{t+1} = \beta_1 m_t + (1 - \beta_1) \mathbf{g}_t, \qquad v_{t+1} = \beta_2 v_t + (1 - \beta_2) \mathbf{g}_t^2$$
  $$\hat{m}_{t+1} = \frac{m_{t+1}}{1 - \beta_1^{t+1}}, \qquad \hat{v}_{t+1} = \frac{v_{t+1}}{1 - \beta_2^{t+1}}, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}_{t+1}$$

- **AdamW**:
  $$\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}_{t+1}$$

- **Damped Newton's Method**:
  $$\theta_{t+1} = \theta_t - \eta (\mathbf{H}_t + \mu \mathbf{I})^{-1} \mathbf{g}_t$$

---

### 4. Filter-Wise Normalized Slices (Li et al., NeurIPS 2018)

When visualizing deep neural network loss surfaces in 2D parameter subspaces:

$$
\mathcal{L}(\alpha, \beta) = L(\theta^* + \alpha \delta + \beta \eta)
$$

random directions $d \in \mathbb{R}^D$ suffer from scale invariance in networks with batch normalization or ReLU activations. `loss-topography` normalizes each weight filter $d_{l, j}$ independently:

$$
\delta_{l, j} = \frac{d_{l, j}}{\Vert d_{l, j} \Vert_F} \Vert \theta^*_{l, j} \Vert_F
$$

This yields scale-invariant comparisons proving why **ResNet skip connections** eliminate chaotic, non-convex fracturing and create convex-like funnels.

---

## 📊 Benchmark Suite Results

Summary results from running `python scripts/run_benchmarks.py`:

| Surface | Optimizer | Steps to Reach $\epsilon$ | Final Loss | Wall-Clock ($\mu\text{s}$/step) | Path Efficiency | Oscillation Metric |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Rosenbrock** | **Damped Newton** | **23** | **0.0000** | 55.9 | 0.2403 | 15.75 |
| Rosenbrock | AdamW | — | 0.7942 | 42.6 | 0.5554 | 0.52 |
| Rosenbrock | Adam | — | 0.9179 | 39.6 | 0.5556 | 0.50 |
| Rosenbrock | AdaGrad | — | 0.6140 | 39.4 | 0.4373 | 1.06 |
| **Rastrigin** | **AdaGrad** | **24** | **0.0000** | 41.9 | 0.2232 | 6.17 |
| Rastrigin | Adam | — | 0.9950 | 43.4 | 0.4409 | 0.29 |
| Rastrigin | Momentum | — | 7.4341 | 50.3 | 0.0344 | 0.27 |
| **Regularized Saddle** | **NAG** | **18** | **-2.5000** | 38.8 | 0.6295 | 0.01 |
| Regularized Saddle | Adam | — | -2.5000 | 40.2 | 0.6705 | 0.01 |
| Regularized Saddle | SGD | — | -2.5000 | 38.4 | 0.9896 | 0.01 |

> **Metric Definitions & Interpretation**:
> - **Steps to Reach $\epsilon$**: Number of iterations required to reach within $\epsilon = 0.01$ of the analytical global optimum $f^*$. An em-dash (`—`) indicates the optimizer did not converge within the maximum step budget (300 steps) due to ill-conditioning or local minima stagnation.
> - **Final Loss on Regularized Saddle**: The Regularized Saddle $f(x, y) = x^2 - y^2 + 0.1x^4 + 0.1y^4$ has global minima at $(0, \pm\sqrt{5})$ with true analytical minimum value $f^* = -2.5000$. Achieving $-2.5000$ indicates successful saddle escape.
> - **Path Efficiency**: Ratio of straight-line Euclidean displacement to total traversed arc length:
>   $$
>   \text{Path Efficiency} = \frac{\Vert \theta_{\text{final}} - \theta_0 \Vert_2}{\sum_{t=0}^{T-1} \Vert \theta_{t+1} - \theta_t \Vert_2} \in (0, 1]
>   $$
>   A value of $1.0$ represents a direct geodesic, while lower values reflect oscillatory wandering or wall bouncing.

---

## 🖥️ Interactive Web Laboratory (Streamlit + Plotly)

Launch the full interactive visualization lab in your browser:

```bash
streamlit run app.py
```

The web application provides five dedicated investigation rooms:

1. **🏁 3D Optimizer Race Arena**:
   - Interactive 3D Plotly surface with camera orbit and pan
   - Live toggling of 7 optimizers simultaneously
   - Real-time parameter sliders (Learning rate $\eta$, Momentum $\beta$, steps $N$, starting coordinates $x_0, y_0$)
   - Trajectory metrics comparison table (Final Loss, Total Path Length, Path Efficiency, Latency)

2. **🔍 Saddle & Curvature Inspector**:
   - Real-time 2D contour vector field displaying gradient arrows $\nabla f$
   - Hessian eigen-decomposition: principal curvature directions visualized as orthogonal eigenvector axes
   - Interactive coordinate picker displaying $\lambda_1, \lambda_2$, determinant $\det(\mathbf{H})$, trace $\text{Tr}(\mathbf{H})$, and critical point classification

3. **📊 Sharp vs. Flat Minima Simulator**:
   - Interactive twin-well potential with adjustable sharpness ratio $k \in [5, 100]$
   - Real-time distribution shift slider ($\Delta \theta \in [-1.0, +1.0]$)
   - Live Generalization Gap metric display comparing test error degradation

4. **📈 Loss Function Sandbox**:
   - Interactive side-by-side plots of loss curves $\mathcal{L}(r)$, first derivative $\nabla\mathcal{L}(r)$, and second derivative $\nabla^2\mathcal{L}(r)$
   - Real-time hyperparameter adjustment (Huber $\delta$, Focal $\gamma$, Quantile $\tau$, Cauchy $c$)
   - Outlier sensitivity tester: observe gradient saturation under extreme residuals

5. **🧠 Real Neural Network Landscape Slicer**:
   - Train a real PyTorch MLP or ResNet on synthetic non-linear classification datasets (Two Moons, Circles)
   - Compute Li et al. filter-normalized random 2D directional slices $(\alpha, \beta)$
   - Compare loss surface smoothness between plain feedforward architectures and skip-connection residual architectures

---

## 🎨 Headless Asset Generation Pipeline

Regenerate all dark-mode GIFs, HTML visualizers, and PNG charts in deterministic headless mode:

```bash
python scripts/generate_readme_assets.py
```

Outputs generated into `assets/`:
- `assets/optimization_river_flow.gif` (Hydrodynamic river flow down non-convex potential)
- `assets/hero_optimizer_race.gif` (Multi-modal 3D/2D synchronized race)
- `assets/interactive_3d_landscape.html` (Standalone 3D orbit WebGL viewer)
- `assets/saddle_point_escape.gif` (Strict saddle negative curvature dynamics)
- `assets/sharp_vs_flat_minima.gif` (Distribution shift & generalization gap animation)
- `assets/loss_function_atlas.png` (High-DPI 10-loss taxonomy panel)
- `assets/loss_convergence_timing.png` (Four-panel convergence, timing, and efficiency dashboard)
- `assets/filter_normalized_nn_surface.png` (Plain MLP vs ResNet 3D + contour landscape)

---

## 📁 Repository Structure

```
loss-topography/
├── app.py                           # Interactive Streamlit + Plotly multi-tab web application
├── assets/                          # Generated high-resolution GIFs, HTML visualizer, and PNG figures
│   ├── filter_normalized_nn_surface.png
│   ├── hero_optimizer_race.gif
│   ├── interactive_3d_landscape.html
│   ├── loss_convergence_timing.png
│   ├── loss_function_atlas.png
│   ├── optimization_river_flow.gif
│   ├── saddle_point_escape.gif
│   └── sharp_vs_flat_minima.gif
├── loss_topography/                 # Modular core library
│   ├── __init__.py                  # Package exports
│   ├── benchmarks.py                # Convergence timing & path efficiency benchmarking
│   ├── losses.py                    # 10 Loss functions with analytical grad & hessian
│   ├── nn_landscape.py              # Filter-normalized PyTorch landscape slicing
│   ├── optimizers.py                # 8 Optimizers with trajectory tracking & timing
│   ├── sharpness.py                 # Sharp vs flat minima & distribution shift simulator
│   ├── surfaces.py                  # 9 Analytical surfaces, critical point classifier
│   └── theme.py                     # Curated dark-mode palette & plotting styles
├── scripts/                         # Executable CLI tools & renderers
│   ├── export_interactive_html.py   # Standalone 3D HTML exporter
│   ├── generate_readme_assets.py    # Master headless asset generation pipeline
│   ├── generate_river_flow.py       # Optimization river flow animation generator
│   ├── quickstart_replicate.py      # 10-second local replication script
│   └── run_benchmarks.py            # Comprehensive benchmark runner
├── tests/                           # Complete test suite (67 unit tests)
│   ├── test_losses.py               # Finite difference verification & properties
│   ├── test_nn_landscape.py         # PyTorch landscape slicing validation
│   ├── test_optimizers.py           # Convergence, trajectories, timing tests
│   ├── test_sharpness.py            # Curvature, shift & sharpness tests
│   └── test_surfaces.py             # Hessian symmetry, eigenvalues, gradients
├── pyproject.toml                   # Modern packaging metadata
├── requirements.txt                 # Pinned dependencies
├── LICENSE                          # MIT License
└── README.md                        # Flagship documentation & visual showcase
```

---

## 🧪 Testing & Validation

The test suite rigorously verifies mathematical correctness using exact analytical formulas and numerical finite differences across all components:

```bash
python -m pytest tests/ -v
```

Validation guarantees:
- **Gradient correctness**: Every loss and surface gradient is checked against central finite differences ($|g_{\text{exact}} - g_{\text{num}}| < 10^{-4}$).
- **Hessian symmetry**: All Hessian matrices verify $\mathbf{H} = \mathbf{H}^T$.
- **Critical point classification**: Eigenvalue signs correctly classify minima, maxima, and saddles.
- **Optimizer trajectory monotonicity**: Convergence on convex quadratics and Rosenbrock valley.
- **Scale invariance**: PyTorch filter normalization verified across weights of differing layer magnitudes.

---

## 📚 Key References

1. **Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T.** (2018). *Visualizing the Loss Landscape of Neural Nets*. Advances in Neural Information Processing Systems (NeurIPS 2018).
2. **Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P.** (2017). *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*. International Conference on Learning Representations (ICLR 2017).
3. **Hochreiter, S., & Schmidhuber, J.** (1997). *Flat Minima*. Neural Computation, 9(1), 1-42.
4. **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017). *Focal Loss for Dense Object Detection*. IEEE International Conference on Computer Vision (ICCV 2017).
5. **Dauphin, Y. N., et al.** (2014). *Identifying and attacking the saddle point problem in high-dimensional non-convex optimization*. Advances in Neural Information Processing Systems (NeurIPS 2014).

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
