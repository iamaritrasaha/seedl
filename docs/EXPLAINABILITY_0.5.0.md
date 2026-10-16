# Advanced Feature Explainability & Attribution Diagnostics

## 1. Overview

Feature attribution in deep learning answers the question: *Which input features most strongly influenced the model's prediction?*

In SeeDL 0.5.0, explainability capabilities have been elevated from basic visualizations into **mathematically grounded, verifiable diagnostic tools**:

1. **SmoothGrad**: Gradient sensitivity averaged over Gaussian noise-perturbed inputs to suppress high-frequency visual noise.
2. **Integrated Gradients with Completeness Diagnostics**: Path-integrated gradients evaluated with the composite midpoint rule, accompanied by exact numerical verification of the Completeness Axiom:
   $$\sum_{i} \text{IG}_i(x) \approx F(x) - F(x_0)$$
3. **Automated Grad-CAM Layer Discovery**: Automatic detection of target convolutional layers with optional user layer overrides.
4. **Cross-Method Attribution Comparison**: Aligned side-by-side evaluation calculating Pearson correlation and Cosine similarity across methods.

---

## 2. Integrated Gradients: Mathematical Invariants

Integrated Gradients satisfies two fundamental axioms of attribution methods:
- **Completeness**: The attributions of all input features sum to the difference between the model's output at the input $x$ and the baseline $x_0$:
  $$\sum_{i=1}^d \text{IG}_i(x) = F(x) - F(x_0)$$
- **Implementation Invariance**: Two functionally identical models assign identical attributions regardless of parameterization.

### Composite Midpoint Rule
In SeeDL, the path integral:
$$\text{IG}_i(x) = (x_i - x_{0,i}) \times \int_{0}^1 \frac{\partial F(x_0 + \alpha(x - x_0))}{\partial x_i} d\alpha$$
is approximated using the **composite midpoint rule** (an open Newton-Cotes formula with uniform weights):
$$\alpha_k = \frac{k - 0.5}{m}, \quad k \in \{1, \dots, m\}$$
$$\text{IG}_i(x) \approx \frac{x_i - x_{0,i}}{m} \sum_{k=1}^m \frac{\partial F(x_0 + \alpha_k(x - x_0))}{\partial x_i}$$

SeeDL computes and reports the **Completeness Error**:
$$\text{Error} = \left| \sum_{i} \text{IG}_i(x) - (F(x) - F(x_0)) \right|$$

```python
viz = sd.explain(
    model,
    input=x,
    target=0,
    method="integrated_gradients",
    steps=50,
)

# Inspect diagnostics
diagnostics = viz.data["diagnostics"]
print(f"Completeness Error: {diagnostics['completeness_error']:.6e}")
print(f"Convergence History: {diagnostics['convergence_curve']}")
```

---

## 3. SmoothGrad

Vanilla gradient saliency often appears noisy because gradients oscillate rapidly in deep non-linear networks. SmoothGrad stabilizes attributions by averaging gradients over $N$ noisy samples:
$$\hat{S}(x) = \frac{1}{N} \sum_{k=1}^N \nabla_x F(x + \epsilon_k), \quad \epsilon_k \sim \mathcal{N}(0, \sigma^2 I)$$
where $\sigma = \text{stdev\_spread} \times (\max(x) - \min(x))$.

```python
viz = sd.explain(
    model,
    input=x,
    target=0,
    method="smoothgrad",
    samples=25,
    stdev_spread=0.15,
    seed=42,  # Reproducible PRNG seed
    abs=False,  # Set to False to preserve signed directions
)
```

---

## 4. Cross-Method Attribution Comparison (`sd.compare_attributions`)

Compare multiple attribution algorithms on the identical model and input with aligned metrics:

```python
comparison = sd.compare_attributions(
    model,
    input=x,
    target=0,
    methods=["saliency", "integrated_gradients", "smoothgrad"],
    steps=20,
    samples=20,
)
```

The output visualization includes:
- Side-by-side heatmaps on identical color scales.
- Pearson correlation coefficient:
  $$r(a, b) = \frac{\sum (a_i - \bar{a})(b_i - \bar{b})}{\sqrt{\sum (a_i - \bar{a})^2 \sum (b_i - \bar{b})^2}}$$
- Cosine similarity:
  $$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}$$
- Disagreement highlights where methods identify divergent explanatory regions.
