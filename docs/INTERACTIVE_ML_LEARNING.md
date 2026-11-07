# Interactive Machine Learning and Deep Learning Lessons

SeeDL v0.7.0 introduces a comprehensive educational simulation system for foundational machine learning and deep learning concepts. Every lesson runs offline, requires zero optional dependencies, and generates interactive SVG and self-contained HTML visual dossiers with bundled Google Sans and JetBrains Mono typography.

---

## Architecture and Design Principles

1. **Pure Python & Zero Dependencies**: All educational modules run using Python's standard library and SeeDL's built-in SVG/HTML renderer. No NumPy, PyTorch, or scikit-learn installation is required to generate or view lessons.
2. **Mathematical Precision**: Simulations compute closed-form analytical solutions, explicit gradient updates, or exact matrix factorizations rather than qualitative approximations.
3. **Data Provenance**: Every simulation embeds declared ground-truth parameters, reproducible seeds, and explicit metric definitions.
4. **Offline Self-Contained Exports**: Output files bundle CSS styles and vector assets. No external CDNs, tracking scripts, or remote web fonts are accessed.

---

## 1. Classical Machine Learning Modules

SeeDL provides 8 dedicated classical machine learning simulations.

### 1.1 Linear Regression Explorer (`linear_regression`)

Explores simple Ordinary Least Squares (OLS) regression, candidate line adjustments, residuals, and goodness-of-fit.

- **Mathematical Formulation**:
  $$\hat{y} = w x + b$$
  $$\text{Loss}_{\text{MSE}}(w, b) = \frac{1}{N} \sum_{i=1}^N (y_i - (w x_i + b))^2$$
  $$\text{Analytical Solution: } w^* = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad b^* = \bar{y} - w^* \bar{x}$$
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
- **Parameters**:
  - `slope` (float, default `1.8`): Slope $w$ of candidate regression line.
  - `intercept` (float, default `0.8`): Intercept $b$ of candidate regression line.
- **Visual Features**:
  - Scatter plot of authentic synthetic data points with vertical residual error bars connecting observations to candidate $\hat{y}$.
  - Side-by-side comparison between candidate line $(w, b)$ and analytical OLS minimum $(w^*, b^*)$.
  - Diagnostics readout: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and coefficient of determination ($R^2$).
- **Reproducible Example**:
  ```python
  import seedl as sd

  view = sd.learn("linear_regression", slope=2.4, intercept=1.1, backend="svg")
  view.save("linear_regression.html")
  ```

---

### 1.2 Polynomial Regression & Overfitting Explorer (`polynomial_regression`)

Visualizes non-linear curve fitting, model capacity, training vs. validation error, and Ridge ($L_2$) regularization.

- **Mathematical Formulation**:
  $$\hat{y} = \sum_{j=0}^d w_j x^j$$
  $$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$
  Where $\mathbf{X}_{i, j} = x_i^j$, $d$ is polynomial degree, and $\alpha \ge 0$ is the $L_2$ regularization penalty.
- **Parameters**:
  - `degree` (int 1–10, default `3`): Polynomial degree $d$.
  - `regularization` (float $\ge 0$, default `0.0`): Ridge $L_2$ penalty parameter $\alpha$.
- **Visual Features**:
  - Training and held-out validation data generated from ground-truth non-linear function $f(x) = \sin(\pi x) + \epsilon$.
  - Continuous fitted polynomial curve plotted against the ground truth function.
  - Multi-degree capacity curve plotting training MSE vs. validation MSE across degrees 1 to 10, clearly highlighting underfitting ($d < 3$) and Runge phenomenon overfitting ($d \ge 7$).
- **Reproducible Example**:
  ```python
  import seedl as sd

  # Low degree (underfitting)
  underfit = sd.learn("polynomial_regression", degree=1)
  # High degree with regularization (controlled capacity)
  regularized = sd.learn("polynomial_regression", degree=8, regularization=0.01)
  regularized.save("polynomial_regularized.html")
  ```

---

### 1.3 Gradient Descent Laboratory (`gradient_descent`)

Demonstrates first-order iterative optimization on convex loss surfaces, step size dynamics, and convergence trajectories.

- **Mathematical Formulation**:
  $$f(w_1, w_2) = \frac{1}{2}(w_1^2 + 4 w_2^2)$$
  $$\nabla f(w_1, w_2) = [w_1, 8 w_2]^T$$
  $$w^{(t+1)} = w^{(t)} - \eta \nabla f(w^{(t)})$$
  Where $\eta > 0$ represents the learning rate.
- **Parameters**:
  - `lr` (float, default `0.15`): Step size / learning rate $\eta$.
  - `steps` (int 1–50, default `10`): Number of optimization iterations.
  - `start_x` (float, default `-3.5`): Initial coordinate $w_1^{(0)}$.
  - `start_y` (float, default `2.5`): Initial coordinate $w_2^{(0)}$.
- **Visual Features**:
  - 2D contour plot of the elliptical quadratic loss surface with level curves.
  - Vector trajectory tracing consecutive iterates from $(w_1^{(0)}, w_2^{(0)})$ toward the global minimum $(0, 0)$.
  - Loss progression curve $f(w^{(t)})$ versus step number $t$.
  - Convergence diagnostics: step-by-step coordinates, gradient norms $\|\nabla f\|_2$, and status indicators (converged, iterating, or diverging/oscillating).
- **Reproducible Example**:
  ```python
  import seedl as sd

  # Stable convergence
  opt = sd.learn("gradient_descent", lr=0.15, steps=12)
  # High learning rate illustrating oscillatory overshoot
  overshoot = sd.learn("gradient_descent", lr=0.26, steps=10)
  opt.save("gradient_descent.html")
  ```

---

### 1.4 Logistic Regression & Decision Boundary (`logistic_regression`)

Explores linear binary classification, sigmoid log-odds mapping, cross-entropy loss, and separation margins.

- **Mathematical Formulation**:
  $$z = \mathbf{w}^T \mathbf{x} + b$$
  $$P(y=1 | \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
  $$\mathcal{L}_{\text{BCE}}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
  $$\text{Decision Boundary: } \mathbf{w}^T \mathbf{x} + b = 0 \iff P(y=1|\mathbf{x}) = 0.5$$
- **Parameters**:
  - `w1` (float, default `1.5`): Weight for feature $x_1$.
  - `w2` (float, default `-1.2`): Weight for feature $x_2$.
  - `bias` (float, default `0.2`): Model bias / intercept $b$.
- **Visual Features**:
  - 2D feature scatter plot showing two linearly separable classes ($y \in \{0, 1\}$).
  - Continuous probability heatmap depicting $\sigma(z) \in [0, 1]$ across the feature plane.
  - Solid decision boundary line where $\hat{p} = 0.5$, plus confidence contours at $\hat{p}=0.25$ and $\hat{p}=0.75$.
  - Binary cross-entropy loss, overall classification accuracy, and false positive / false negative breakdown.
- **Reproducible Example**:
  ```python
  import seedl as sd

  view = sd.learn("logistic_regression", w1=1.8, w2=-1.4, bias=0.1)
  view.save("logistic_regression.html")
  ```

---

### 1.5 K-Nearest Neighbors Classifier (`knn`)

Demonstrates instance-based learning, distance metric geometry, and local decision surfaces.

- **Mathematical Formulation**:
  $$\text{Dist}_p(\mathbf{x}, \mathbf{x}_i) = \left( \sum_{j=1}^D |x_j - x_{i,j}|^p \right)^{1/p}$$
  $$\mathcal{N}_k(\mathbf{x}) = \text{argmin}_{S \subset D, |S|=k} \sum_{\mathbf{x}_i \in S} \text{Dist}(\mathbf{x}, \mathbf{x}_i)$$
  $$\hat{P}(y = c | \mathbf{x}) = \frac{1}{k} \sum_{\mathbf{x}_i \in \mathcal{N}_k(\mathbf{x})} \mathbb{I}(y_i = c)$$
- **Parameters**:
  - `k` (int 1–15, default `3`): Number of nearest neighbors to query.
  - `metric` (str: `"euclidean"` or `"manhattan"`, default `"euclidean"`): Distance metric $L_2$ ($p=2$) vs. $L_1$ ($p=1$).
  - `query_x` (float, default `0.5`): First coordinate of test query point.
  - `query_y` (float, default `0.5`): Second coordinate of test query point.
- **Visual Features**:
  - Multi-class 2D dataset with distinct spatial clusters.
  - Highlighted query point surrounded by a bounding radius circle enclosing the $k$ selected neighbors.
  - Direct connecting distance vectors showing neighbor ranks, individual distances, and class labels.
  - Class voting breakdown and predicted class probability distribution.
- **Reproducible Example**:
  ```python
  import seedl as sd

  view = sd.learn("knn", k=5, metric="euclidean", query_x=0.2, query_y=-0.3)
  view.save("knn.html")
  ```

---

### 1.6 Decision Tree Learning Explorer (`decision_tree`)

Illustrates recursive axis-aligned feature space partitioning, impurity criteria, and hierarchical tree node structures.

- **Mathematical Formulation**:
  $$\text{Gini}(S) = 1 - \sum_{c=1}^C p_c^2$$
  $$\text{Entropy}(S) = -\sum_{c=1}^C p_c \log_2(p_c)$$
  $$\Delta I(S, A) = I(S) - \frac{|S_L|}{|S|} I(S_L) - \frac{|S_R|}{|S|} I(S_R)$$
- **Parameters**:
  - `max_depth` (int 1–4, default `2`): Maximum tree depth.
  - `criterion` (str: `"gini"` or `"entropy"`, default `"gini"`): Impurity evaluation metric.
- **Visual Features**:
  - 2D partitioned feature space with axis-aligned bounding boxes colored by dominant leaf prediction.
  - Structured decision tree graph displaying split thresholds (e.g., $x_1 \le \theta$), sample counts $N$, and impurity values at every internal node and leaf.
  - Impurity reduction gain metrics and leaf purity percentages.
- **Reproducible Example**:
  ```python
  import seedl as sd

  tree_view = sd.learn("decision_tree", max_depth=3, criterion="gini")
  tree_view.save("decision_tree.html")
  ```

---

### 1.7 K-Means Clustering Explorer (`kmeans`)

Visualizes Lloyd's iterative algorithm, centroid relocations, Voronoi cell partitioning, and within-cluster variance (inertia).

- **Mathematical Formulation**:
  $$\text{Assignment: } c_i^{(t)} = \text{argmin}_k \|\mathbf{x}_i - \boldsymbol{\mu}_k^{(t)}\|^2$$
  $$\text{Update: } \boldsymbol{\mu}_k^{(t+1)} = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{x}_i$$
  $$\text{Inertia (WCSS)} = \sum_{k=1}^K \sum_{i \in S_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$
- **Parameters**:
  - `k` (int 2–5, default `3`): Number of clusters.
  - `iterations` (int 1–10, default `3`): Number of Lloyd update steps executed.
  - `seed` (int, default `42`): Random seed for initial centroid placement.
- **Visual Features**:
  - Scatter plot with observations colored by assigned cluster centroid.
  - Prominent cluster centroids ($\boldsymbol{\mu}_k$) marked with crosshairs and centroid history trajectory paths.
  - Voronoi cell boundaries delineating assignment regions in feature space.
  - Total inertia readout and per-cluster sample counts and dispersion radii.
- **Reproducible Example**:
  ```python
  import seedl as sd

  clusters = sd.learn("kmeans", k=3, iterations=5, seed=42)
  clusters.save("kmeans.html")
  ```

---

### 1.8 Principal Component Analysis Explorer (`pca`)

Visualizes orthogonal variance maximization, covariance matrix eigendecomposition, and dimensional projection.

- **Mathematical Formulation**:
  $$\boldsymbol{\Sigma} = \frac{1}{N} \mathbf{X}_c^T \mathbf{X}_c$$
  $$\boldsymbol{\Sigma} \mathbf{v}_k = \lambda_k \mathbf{v}_k, \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D$$
  $$\text{Explained Variance Ratio } r_k = \frac{\lambda_k}{\sum_{j=1}^D \lambda_j}$$
  $$\mathbf{z}_i = \mathbf{x}_{c, i} \mathbf{V}_k$$
- **Parameters**:
  - `n_components` (int 1–2, default `2`): Number of principal directions to retain.
- **Visual Features**:
  - Bivariate correlated data cloud centered at origin $(\bar{x}_1, \bar{x}_2) = (0, 0)$.
  - Principal component eigenvectors ($\mathbf{v}_1, \mathbf{v}_2$) rendered as scaled orthogonal arrows whose lengths correspond to eigenvalues $\sqrt{\lambda_k}$.
  - Orthogonal projection lines showing how 2D data points map onto the first principal axis.
  - Scree plot and explained variance percentage bar charts ($r_1$ vs. $r_2$).
- **Reproducible Example**:
  ```python
  import seedl as sd

  pca_view = sd.learn("pca", n_components=2)
  pca_view.save("pca.html")
  ```

---

## 2. Deep Learning Educational Modules

In addition to classical machine learning, SeeDL provides 7 core deep learning simulations:

| Concept Slug | Title | Key Equations & Focus |
| --- | --- | --- |
| `backpropagation` | MLP Analytical Chain Rule | Multivariate chain rule $\frac{\partial L}{\partial w_{ij}} = \delta_j a_i$, forward activations and backward error propagation across 3-layer network graph. |
| `activation_functions` | Activation Curves & Derivatives | $f(x)$ and $f'(x)$ comparison for ReLU, Sigmoid, Tanh, and GELU. Explores saturation regimes and vanishing gradients. |
| `loss_functions` | Loss Profiles & Gradients | MSE vs. MAE vs. Huber Loss ($L_\delta(e)$). Analyzes outlier sensitivity and gradient continuity around error zero. |
| `attention` | Scaled Dot-Product Attention | Attention matrix $A = \text{softmax}(QK^T / \sqrt{d_k})$. Query-key dot products, scaling factor $\sqrt{d_k}$, and token distribution weights. |
| `dropout` | Inverted Dropout Simulation | Random Bernoulli masking matrix $m \sim \text{Bernoulli}(1-p)$, active activation scaling by $\frac{1}{1-p}$, and test-time deterministic pass. |
| `batch_norm` | Mini-Batch Standardization | Mean $\mu_B$, variance $\sigma_B^2$, normalization $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$, and affine learnable parameters $\gamma \hat{x} + \beta$. |
| `convolution` | 2D Spatial Kernel Feature Mapping | $2D$ cross-correlation operation $(I * K)_{i, j}$, receptive field tracing, strides, zero padding, and output feature map generation. |

---

## 3. Terminal CLI Usage

All educational simulations can be discovered and exported directly from your terminal:

```bash
# Browse all 15 educational concepts grouped by domain
seedl learn list

# Export linear regression lesson to default linear_regression.html
seedl learn linear_regression

# Export PCA lesson to custom SVG
seedl learn pca --output artifacts/pca_explorer.svg

# Read documentation and parameter details
seedl help learn
```

---

## 4. Scientific Rigor and Disclosures

- **Ground Truth Integrity**: Synthetic educational datasets use fixed seeds and disclosed equations. They demonstrate algorithm mechanics rather than clinical or financial predictive claims.
- **Numerical Stability**: Ridge solvers and Gaussian elimination routines include threshold pivoting ($\epsilon = 10^{-12}$) to guarantee finite real computations.
- **Offline Typography Guarantee**: Every export embeds OFL-licensed Google Sans and JetBrains Mono fonts directly in base64 data URIs within the SVG `<style>` tag, ensuring pixel-perfect fidelity without internet access.
