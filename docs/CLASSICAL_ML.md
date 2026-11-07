# Classical machine learning

SeeDL inspects and evaluates scikit-learn models. Train models in your own
scikit-learn code; inspection/evaluation never calls `fit()`.

These APIs and the `sklearn` extra are available in SeeDL 0.7.0:


```bash
python -m pip install "seedl[sklearn]"
seedl help evaluate
seedl examples evaluate
seedl doctor
```

The optional extra installs `scikit-learn>=1.4.2,<2` and its dependencies. The
resolver selects a compatible release for your Python version; the SeeDL core
still has zero mandatory dependencies. Pandas is supported when you already
use DataFrames; it is not required by the adapter. SVG and offline HTML require
no optional plotting backend. Plotly and Matplotlib remain optional.

## Start with one evaluation call

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seedl as sd

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
result = sd.evaluate(model, X_test, y_test, split="test", backend="svg")
result.show()
result.save("evaluation.html")
```

The fitting above is ordinary sklearn code. The evaluation contains metrics,
class support, and selectable captured plots:

```python
print(result.metrics)
print(result.available_plots)
result.plot("confusion_matrix").save("confusion.svg")
print(result.unavailable_plots)
```

Selecting, displaying, and saving a captured view never predicts again. The
result is a `Visualization` subclass (`Evaluation`) with the existing `.show()`,
`.save()`, `.to_text()`, `.to_html()`, `.data`, `.metadata`, and `.notes` contract.
In notebooks, return `result` as the final expression or use `.show()`.

`backend="auto"` keeps SeeDL's existing choice: Plotly if installed, otherwise
SVG. The document uses the shared SeeDL style and bundled Google Sans /
JetBrains Mono fonts. No browser is opened by evaluation or export. An explicit
terminal `.show()` with Plotly/Matplotlib keeps that backend's display behavior.

## Supervised metrics

| Task | Metrics and captured plots |
| --- | --- |
| Classification | Accuracy; per-class precision, recall, F1 and support; macro/weighted metrics; confusion matrix with class names and optional `normalize="true"`, `"pred"`, or `"all"` |
| Binary score diagnostics | ROC-AUC and precision-recall curve with average precision, using valid `predict_proba` or `decision_function` scores |
| Regression | MAE, MSE, RMSE, R²; actual versus predicted scatter; residual versus predicted; residual distribution |

Example selection at capture time:

```python
result = sd.evaluate(
    model,
    X_test,
    y_test,
    split="test",
    plots="confusion_matrix",
    normalize="true",
    class_names={0: "setosa", 1: "versicolor", 2: "virginica"},
    backend="svg",
)
```

`class_names` may be a label-to-name mapping or a list in sorted observed/model
label order. Unseen true classes are included in the report instead of being
silently discarded. Names must be unique. A binary `positive_label=` defaults
to `model.classes_[1]`; choose it explicitly when numeric ordering does not
match the question (for the Breast Cancer dataset, 0 means malignant).

A decision function is labeled as an uncalibrated margin, not a probability.
Missing probabilities are never reconstructed from predicted labels. Binary
ROC/PR requires two fitted classes, both represented in the reference targets.
Multiclass evaluation returns per-class and aggregate metrics; binary curves
are unavailable with an explicit explanation. It does not silently choose a
one-vs-rest class. Average precision is not trapezoidal PR area.

Undefined precision/recall/F1 values remain unavailable; sklearn excludes them
from the corresponding macro/weighted averages. Confusion-matrix normalization
uses sklearn's documented zero convention for empty denominators. R² uses
`force_finite=False`: constant targets or fewer than two samples can make it
undefined. Negative R² is preserved. Numerical overflow/underflow is disclosed;
missing or invalid statistics are never replaced with invented values.

Regression axes use the target's units; SeeDL cannot infer physical units from
a numeric array. Record them in notes/metadata, as the California Housing
notebook does. MSE has squared target units, MAE/RMSE have target units, and R² is
dimensionless. Residuals mean **actual minus predicted**.

## Read-only estimator inspection

```python
inspection = sd.see(model, backend="svg")
inspection.save("model.html")
# Additional bounded tree/feature controls, without changing see() parameters:
inspection = sd.inspect_estimator(model, max_depth=3, max_nodes=63, max_features=50)
```

Inspect fitted status, estimator type, constructor settings, feature counts,
input names, linear/logistic coefficients and intercepts, decision-tree splits,
forest importance, pipeline structure, fitted PCA variance or existing search
results where available. Unfitted models are inspectable without fitting them.
Previously defined tensor/PyTorch/quantum `see()` arguments keep their meanings;
use `evaluate()` rather than passing sklearn evaluation data as `see(input=...)`.

Pipeline diagrams distinguish sequential steps from parallel ColumnTransformer
branches. Original input column selectors and transformed coefficient names are
separate. Polynomial, one-hot and other expanded names are used only when fitted
preprocessing provides `get_feature_names_out()` reliably. Otherwise coefficient
labels explicitly say `transformed feature N`; original names are not substituted.

Coefficients depend on feature scaling/encoding and are not causal effects.
Binary logistic coefficients refer to `classes_[1]` log-odds even when an
**evaluation** selects the other positive label. Tree thresholds and child IDs
are preserved in structured data; tree previews disclose depth/node bounds.

## Importance, partial dependence and ICE

```python
native = sd.feature_importance(model, backend="svg")
permutation = sd.feature_importance(
    model,
    X_test,
    y_test,
    method="permutation",
    split="test",
    scoring="accuracy",
    n_repeats=5,
    max_samples=512,
    backend="svg",
)
```

Native displays identify impurity-based importance or signed fitted coefficients.
Impurity importance can favor high-cardinality features. Permutation importance
uses explicitly supplied evaluation data and permutes **original input columns
through the whole pipeline**. It reports mean score decrease and repeat standard
deviation, preserving negative values. It is not a causal estimate or confidence
interval. Scorers with negative loss conventions retain their sign.

```python
# This Iris example is multiclass, so choose one response class explicitly:
response = sd.partial_dependence(
    model,
    X_test,
    [0],
    kind="both",
    target=1,
    grid_resolution=20,
    max_samples=128,
    max_ice=8,
    backend="svg",
)
# Multiclass needs target=<one fitted class label>.
```

PDP/ICE call sklearn's brute-force implementation without fitting. Features are
one to four numeric **original input** columns, selected by index/name. Reference
rows, percentile grid, ICE limits and response units are disclosed. Varying a
correlated feature may create unrealistic combinations; these are model-response
summaries, not causal interventions. Binary responses follow sklearn's
`classes_[1]` convention; multiclass requires an explicit target class label.

## Clusters and fitted PCA

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

clusterer = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X_train)
projection = PCA(n_components=2).fit(X_train)  # your explicit optional fitting
clusters = sd.evaluate(clusterer, X_test, projection=projection, split="test")

# DBSCAN cannot assign clusters to new rows; training labels need explicit alignment.
dbscan = DBSCAN().fit(X_train)
training_clusters = sd.evaluate(
    dbscan, X_train, labels=dbscan.labels_, split="training", backend="svg"
)
pca_view = sd.evaluate(projection, X_test, split="test", backend="svg")
```

Without an explicit fitted PCA, cluster plots show the first one/two estimator
feature coordinates (including fitted preprocessing for pipelines). No PCA is
fit automatically. Cluster-size summaries use all supplied rows and include
noise (-1). PCA projections disclose omitted components; explained variance
comes from PCA's fitting population, not a new fit on the evaluation rows.

Silhouette uses sklearn in the clusterer's actual feature space, with its
distance metric, on a bounded non-noise sample (at most 1,000 rows). It does not
use display PCA coordinates. Noise exclusion and sampled rows are disclosed.
Fewer than two clusters, singleton-only assignments, or an invalid bounded
sample produce an unavailable result. It is separation evidence, not external
classification accuracy. DBSCAN training-row identity cannot be recovered from
`labels_`; matching lengths alone do not prove alignment.

## Existing model-selection results and explicit extra training

```python
# search was fitted by your code; scores/folds were already computed:
# sd.cv_results(search, metric="score").save("search.html")
# sd.cv_results(cross_validate_results).save("folds.html")
```

`sd.see(fitted_grid_search)` also previews existing results. Multi-metric searches
can be selected using `sd.cv_results(search, metric="accuracy")`. CV held-out
folds are labeled validation, not an untouched final test set. Failed scores
remain unavailable; they are not dropped or replaced with zero. Candidate
parameter previews and score standard deviations are labeled and bounded.

The following operations necessarily train estimator **clones**, so they fail
without an explicit `allow_refit=True`:

```python
curve = sd.learning_curve(
    model,
    X_train,
    y_train,
    allow_refit=True,
    cv=3,
    train_sizes=[0.5, 1.0],
    max_samples=1000,
    backend="svg",
)
validation = sd.validation_curve(
    model,
    X_train,
    y_train,
    param_name="C",
    param_range=[0.1, 1.0, 10.0],
    allow_refit=True,
    cv=3,
    backend="svg",
)
```

Use training data for these model-selection diagnostics. The original estimator
is unchanged. Bounds are five folds, six settings, one job, and 5,000 reference
rows; estimator internals can fit submodels. Nested searches are rejected; use an
explicitly selected `search.best_estimator_` instead.

## Existing SeeDL APIs remain useful

- `sd.predictions(scores, targets=..., ...)` presents already computed scores/predictions. It is unchanged; it does not inspect or fit sklearn models.
- `sd.compare_predictions(...)` compares supplied score vectors. It is unchanged and does not replace evaluation on aligned labeled datasets.
- `sd.dataset_health(X)` remains the independent data-health workflow; the regression notebook combines it with the new evaluator.
- `sd.report(views=[result, ...])` combines captured views using the existing offline report system.
- `sd.save_analysis(result, ...)` preserves the captured JSON payload. `load_analysis()` returns the existing general `Visualization` type; metrics/panels remain in `.data` and render without requiring sklearn.

The adapter delegates model metrics to sklearn rather than reusing the old
single-sample score formatter, which can normalize logits. That avoids treating
labels as probabilities or introducing competing metric formulas.

## Safety and limitations

- Supervised targets must be single-output, finite and row-aligned. Regression targets/predictions must be real; complex values are not silently cast. NumPy uses positional alignment; pandas indices and sklearn feature-name validation catch reordered/mismatched frames.
- CPU NumPy arrays, pandas DataFrames and relevant SciPy sparse matrices are accepted. Framework/GPU tensors require explicit caller conversion; existing PyTorch inspection is unchanged. Input data is copied to protect in-place preprocessing; original sparse feature matrices are not densified. Small two-column plot previews and bounded pairwise distance matrices may be dense.
- Default plot previews are 2,048 rows. Evaluation is capped at 1,000,000 rows and 10,000,000 dense/stored input values; known transformed-width expansion is checked before prediction. Metrics use all supplied rows unless a specific sampled diagnostic says otherwise.
- Classification displays at most 100 classes. Tree previews, parameters, coefficients, candidates and feature names have disclosed limits. Native importance is omitted beyond 1,000 ensemble members or 100,000 fitted features.
- Permutation importance is bounded to 200 input features, 2,000 reference rows and 20 repeats; sparse combinations not supported by sklearn fail clearly rather than silently densifying.
- PDP supports numeric individual features, not interaction surfaces or categorical grids; at most 512 reference rows, 50 grid points and 50 displayed ICE curves per feature. Classification supports 2–20 fitted classes.
- Multi-output/multilabel supervised evaluation and multiclass ROC/PR are intentionally unsupported. Precomputed-distance clustering is inspectable but its feature-space evaluation is unsupported.
- Standard sklearn estimators/pipelines are tested for state preservation. Arbitrary custom predictor/transformer/scorer side effects are not sandboxed; SeeDL does not promise to undo arbitrary user code or bound its internals.
- Metrics and plots do not prove that data were held out, prevent leakage, establish causality, or validate a model for deployment.

## Executable examples and evidence

- [Offline demo script](../examples/06_classical_ml.py): bundled Breast Cancer and Iris data; explicit caller training, no downloads.
- [California Housing regression notebook](../examples/classical_ml_california_housing.ipynb): authentic data, units, held-out errors, interpretation and report.
- [Breast Cancer classification notebook](../examples/classical_ml_breast_cancer.ipynb): explicit malignant positive class, ROC/PR and pipeline inspection.
- [Compatibility, screenshots and actual validation](CLASSICAL_ML_COMPATIBILITY.md).
