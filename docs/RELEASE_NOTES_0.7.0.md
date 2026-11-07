# SeeDL 0.7.0 — Classical Machine Learning, Interactive Learning Simulations & Developer Experience

> **Evaluate and interpret scikit-learn estimators.**  
> **Master core ML and deep learning algorithms visually.**  
> **Zero mandatory runtime dependencies. Offline by default.**

**Status:** Published.

---

## 1. Introduction

SeeDL 0.7.0 is a milestone release expanding SeeDL from a deep-learning and quantum visualization toolkit into a unified scientific diagnostic suite for the entire machine learning spectrum.

Version 0.7.0 introduces three major pillars:
1. **Interactive Machine Learning & Deep Learning Educational System**: 15 offline, mathematically verified visual simulations spanning classical ML (`linear_regression`, `polynomial_regression`, `gradient_descent`, `logistic_regression`, `knn`, `decision_tree`, `kmeans`, `pca`) and deep learning (`backpropagation`, `activation_functions`, `loss_functions`, `attention`, `dropout`, `batch_norm`, `convolution`).
2. **Classical Machine Learning Evaluation & Interpretation**: Comprehensive diagnostics for scikit-learn estimators, pipelines, ensembles, clusterers, and regressors (`sd.evaluate`, `sd.feature_importance`, `sd.partial_dependence`, `sd.learning_curve`, `sd.cv_results`) with zero estimator side effects (`fit` is never called).
3. **Developer Experience & Offline Typography**: Complete CLI discoverability (`seedl help`, `seedl list`, `seedl learn`, `seedl doctor`, `seedl examples`), enhanced terminal formatting, Windows console safety (CP1252 encoding guards), and 100% offline self-contained exports powered by embedded Google Sans and JetBrains Mono typography.

As always, SeeDL preserves its core scientific guarantees:
- **Zero Mandatory Runtime Dependencies**: The core package runs on standard Python 3.10–3.13 with zero external dependencies. SVG and HTML generation use pure Python standard libraries.
- **Scientific Integrity & Provenance**: Explicit ground-truth definitions, disclosed seeds, analytical closed-form baselines, and unambiguous labeling of data splits and test boundaries.
- **Non-Destructive Observation**: Estimators and models are inspected without state mutation. Hooks and training modes are restored with guaranteed cleanup in `try...finally` blocks.

---

## 2. Release Highlights

### 2.1 Interactive Machine Learning Learning System (`sd.learn`)
- **15 Mathematical Educational Modules**:
  - **Classical Machine Learning**:
    - `linear_regression`: Ordinary least squares regression, analytical normal equations, residuals, $R^2$, and parameter sensitivity.
    - `polynomial_regression`: Polynomial basis expansion, degree capacity sweep (1–10), underfitting vs. overfitting, and $L_2$ Ridge regularization.
    - `gradient_descent`: First-order gradient updates on quadratic loss surfaces, step-size dynamics, oscillation, and convergence paths.
    - `logistic_regression`: Binary classification, sigmoid link function, log-loss / cross-entropy, and decision boundary geometry.
    - `knn`: K-nearest neighbors instance classification, distance metrics (Euclidean, Manhattan), and local neighborhood voting.
    - `decision_tree`: Recursive binary feature space partitioning, Gini impurity, Information Gain, and tree hierarchy.
    - `kmeans`: Lloyd's algorithm, iterative centroid updates, Voronoi cell partitioning, and within-cluster sum of squares (inertia).
    - `pca`: Sample covariance matrix eigendecomposition, explained variance ratios, and orthogonal subspace projection.
  - **Deep Learning**:
    - `backpropagation`: Multi-layer perceptron chain rule gradient flow.
    - `activation_functions`: ReLU, Sigmoid, Tanh, and GELU function and derivative profiles.
    - `loss_functions`: MSE, MAE, and Huber loss outlier sensitivity comparisons.
    - `attention`: Scaled dot-product attention step-by-step matrix derivation.
    - `dropout`: Inverted dropout masking and activation scaling dynamics.
    - `batch_norm`: Mini-batch standardization and affine transformation.
    - `convolution`: 2D spatial kernel sliding, strides, padding, and feature map creation.
- **CLI Discovery and Direct Export**:
  - `seedl learn list`: Formatted catalog of all lessons grouped by category.
  - `seedl learn <concept> [--output FILE]`: Directly export standalone SVG or offline HTML without writing Python code.

### 2.2 Scikit-Learn Estimator Diagnostics (`sd.evaluate`)
- **Comprehensive Evaluation Dossiers**:
  - Classification: Confusion matrices, ROC curves, Precision-Recall curves, per-class metrics (accuracy, precision, recall, F1, AUC), and decision boundaries.
  - Regression: Residuals vs. fitted values, predicted vs. actual scatter, error distribution histograms, $R^2$, MSE, RMSE, and MAE.
  - Clustering: PCA 2D projections with cluster coloring, Voronoi centroid assignments, silhouette analysis, and inertia metrics.
- **Model Interpretation & Validation**:
  - `sd.inspect_estimator`: Pipeline step visualization, hyperparameter auditing, and feature name resolution.
  - `sd.feature_importance`: Permutation importance and tree-based Gini importance with confidence intervals.
  - `sd.partial_dependence`: One-way and two-way partial dependence functions.
  - `sd.learning_curve` & `sd.validation_curve`: Sample efficiency and hyperparameter sensitivity curves.
  - `sd.cv_results`: Cross-validation search space heatmaps for `GridSearchCV` and `RandomizedSearchCV`.
- **Zero Side-Effects Contract**:
  - SeeDL never invokes `.fit()` or alters fitted estimator parameters (`coef_`, `intercept_`, `tree_`).

### 2.3 Developer Experience & Offline Typography
- **Command-Line Interface Overhaul**:
  - `seedl`: Fast compact command guide.
  - `seedl help <api>`: Real-time signature, docstring, and runnable snippet.
  - `seedl doctor`: Environment audit, Python version checks (3.10–3.13), and optional integration availability without importing heavy frameworks.
  - `seedl examples [topic]`: Print copyable Python snippets.
  - `seedl demo`: Generate authentic labeled UCI Iris demonstration without network calls.
- **Console & Windows Safety**:
  - Graceful ASCII fallback for symbols when running in non-UTF-8 terminals (e.g., CP1252 on legacy Windows command prompts), eliminating `UnicodeEncodeError`.
  - Respects `NO_COLOR` and `--plain` flags for clean CI logs.
- **Typography & Font Embedding**:
  - Bundled Google Sans and JetBrains Mono WOFF2 font assets.
  - Embedded as base64 data URIs inside `<style>` blocks in SVG and HTML exports, guaranteeing complete offline fidelity without CDN calls.

---

## 3. Quickstart Examples

### 3.1 Interactive ML Learning
```python
import seedl as sd

# Explore ordinary least squares regression
view = sd.learn("linear_regression", slope=2.2, intercept=1.0)
view.show()
view.save("linear_regression.html")

# Explore gradient descent optimization steps
gd = sd.learn("gradient_descent", lr=0.15, steps=10)
gd.save("gradient_descent.svg")
```

### 3.2 Classical ML Evaluation
```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seedl as sd

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Evaluate test set
result = sd.evaluate(model, X_test, y_test, split="test")
result.show()
result.save("evaluation.html")
```

---

## 4. Migration from v0.6.0

SeeDL 0.7.0 is **100% backward-compatible** with all v0.6.0 APIs. No existing public functions, arguments, or methods were removed or broken.

### Installation Options
```bash
# Core library (pure Python, zero dependencies)
pip install --upgrade seedl

# Install with scikit-learn integration
pip install "seedl[sklearn]"

# Install all optional integrations (sklearn, PyTorch, NumPy, Plotly, Qiskit)
pip install "seedl[all]"
```

### Upgrading Code
Existing calls to `sd.see`, `sd.activations`, `sd.trace`, `sd.flow`, `sd.watch`, `sd.gradient_health`, and `sd.bloch` continue to work unchanged. To take advantage of v0.7.0 features:
- Use `sd.evaluate(model, X, y)` for your fitted scikit-learn estimators.
- Use `sd.learn(concept)` to generate interactive educational modules for teaching, learning, or documentation.
- Use the `seedl` CLI tools for rapid exploration and diagnostics directly from the command line.
