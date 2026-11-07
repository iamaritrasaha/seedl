# Classical ML implementation and compatibility report

Validated on 2026-11-07 for SeeDL 0.7.0. It builds directly on main `c9b59ba`,
 which contains the developer
experience overhaul from PR #24.

## Baseline and implementation decisions

The baseline was exercised with a fitted `LogisticRegression` on authentic Iris
data. `sd.evaluate` did not exist; `sd.see(model, backend="svg")` raised:

```text
UnsupportedObjectError: Cannot visualize LogisticRegression. SeeDL supports
PyTorch tensors and modules, NumPy arrays, real numbers, numeric sequences,
and metric dictionaries; install optional dependencies for PyTorch or NumPy objects.
```

Existing `predictions`, `compare_predictions`, and `dataset_health` already
cover tensor/probability predictions and data diagnostics. They remain useful
and unchanged. Their input contracts do not replace estimator detection,
sklearn label conventions, fitted pipeline provenance, or sklearn's metric
implementations. The adapter uses those established metric implementations
and the existing SeeDL rendering/export/serialization system.

| Concern | Implementation |
| --- | --- |
| Lazy detection and validation | `_sklearn_detect.py`, `_sklearn_validation.py`: sklearn MRO detection without importing sklearn for normal help/core usage; fitted checks, alignment, feature and resource budgets, defensive data copies |
| Estimator metadata | `_sklearn_metadata.py`: bounded fitted attributes, coefficients, tree children/splits, impurity importance and reliable transformed names |
| Scientific calculations | `_sklearn_metrics.py`, `_sklearn_interpret.py`, `_sklearn_unsupervised.py`, `_sklearn_selection.py`: sklearn metrics/inspection/model-selection functions, explicit sampling and feature-space provenance |
| Capture and public API | `_sklearn.py`, `evaluation.py`, `api.py`, `__init__.py`: `Evaluation(Visualization)` and captured backend-neutral panels; no retained estimator or deferred prediction |
| Presentation | `_ml_renderers.py` extends shared SVG/Plotly/Matplotlib dispatch; shared HTML shell, fonts, text formatter, export and report infrastructure |
| Discovery and examples | Existing CLI/catalog extended; `CLASSICAL_ML.md`, two executable notebooks, bundled-data demo and browser verification script |

Implementation proceeded through tested supervised evaluation, then tested
interpretability/pipelines, then tested clustering/PCA/model selection. There
is no replacement computation engine or required frontend/server.

## Before and after

Previously, users had to extract predictions, select sklearn metrics, assemble
plots, and arrange exports themselves. SeeDL could not inspect their estimator.
After fitting in ordinary sklearn code:

```python
view = sd.see(model, backend="svg")
result = sd.evaluate(model, X_test, y_test, split="test", backend="svg")
result.show()
result.save("evaluation.html")
result.plot("confusion_matrix").save("confusion.svg")
```

The bundled Breast Cancer demo uses a stratified split (`random_state=42`),
`StandardScaler` and `LogisticRegression`. Actual captured results:

```text
143 test samples | 4 captured plots | scikit-learn diagnostics
accuracy           0.986013986013986
precision_macro    0.9850104821802935
recall_macro       0.9850104821802935
f1_macro           0.9850104821802935
roc_auc            0.9976939203354298
average_precision  0.9967570754716981
```

The report includes the 53 malignant / 90 benign class supports and confusion
matrix `[[52, 1], [1, 89]]`. ROC/PR explicitly use positive label 0 (malignant).
This is an educational benchmark, not clinical validation.

The California Housing notebook evaluates a held-out 4,128-row test set:
MAE 0.5332001304956565, MSE 0.5558915986952438,
RMSE 0.745581383012776, R² 0.5757877060324512.
Target units are 100,000 USD; the source has a target cap and historical 1990
census provenance. Scatter previews disclose 2,048 of 4,128 rows; metrics and
the residual histogram use every supplied row.

There is no baseline ML chart to screenshot because main did not support
estimator evaluation. The images below are actual Chromium screenshots of the
new offline exports, rather than mockups:

| View | Desktop | Phone |
| --- | --- | --- |
| Breast Cancer classification | [1280 px](classical-ml-visuals/classification-1280.png) | [390 px](classical-ml-visuals/classification-390.png) |
| California Housing regression | [1280 px](classical-ml-visuals/regression-1280.png) | [390 px](classical-ml-visuals/regression-390.png) |
| Pipeline and column transformations | [1280 px](classical-ml-visuals/pipeline-1280.png) | — |
| Permutation importance | [1280 px](classical-ml-visuals/permutation-1280.png) | — |
| Fitted PCA cluster projection | [1280 px](classical-ml-visuals/clusters-1280.png) | — |
| Partial dependence and ICE | [1280 px](classical-ml-visuals/dependence-1280.png) | — |

The document uses bundled Google Sans and JetBrains Mono. On narrow screens,
chart regions scroll independently to keep SVG labels legible; metric cards
wrap and tables provide keyboard-accessible scrolling. Text alternatives,
printed matrix values and full-precision captured JSON remain available.

## Reproduce the demo and visual checks

```bash
python -m pip install -e ".[sklearn]"
python examples/06_classical_ml.py --output artifacts/classical-ml
seedl help evaluate
seedl examples evaluate
```

The demo uses sklearn's bundled Breast Cancer and Iris datasets without a
download. Open `classification.html` yourself. For all ten acceptance views:

```bash
# pandas and Playwright are verification tools, not SeeDL runtime dependencies.
python -m pip install pandas playwright
python -m playwright install chromium
python scripts/verify_sklearn_visuals.py --screenshots --output artifacts/ml-check
# Explicitly fetch/cache authentic California Housing instead of bundled Diabetes:
python scripts/verify_sklearn_visuals.py --screenshots --california \
  --data-home artifacts/datasets --output artifacts/ml-california
```

The script fits demonstration models explicitly. It checks classifications,
regression, ROC, a ColumnTransformer pipeline, permutation importance, PDP/ICE,
KMeans, DBSCAN, PCA and existing search results at 320, 390, 768 and 1280 px.
All 40 SVG-backed view/viewport checks passed without page overflow, external
HTTP requests or JavaScript errors. Separate automated browser tests exercise
both SVG and Plotly, malicious labels, font loading, metric cards, captured
JSON, serialization and reports.

The [California Housing notebook](../examples/classical_ml_california_housing.ipynb)
and [Breast Cancer notebook](../examples/classical_ml_breast_cancer.ipynb)
were both executed using a real Jupyter kernel, including their HTML/SVG exports.
Source notebooks keep outputs cleared. The California download happens only
at its explicit `fetch_california_housing` call, and is cached thereafter.

## Compatibility and verification

The optional adapter requires `scikit-learn>=1.4.2,<2`; Python support remains
3.10–3.13. No mandatory runtime dependency was added. PyPI metadata was checked:
current stable sklearn 1.9.1 requires Python >=3.11, so Python 3.10 resolves
1.7.2. Isolated environments were tested as follows:

| Python | scikit-learn | NumPy | SciPy | pandas | Adapter + CLI tests |
| --- | --- | --- | --- | --- | --- |
| 3.10.22 | 1.4.2 (minimum) | 2.2.6 | 1.15.3 | 2.3.3 | 102 passed, 6 skipped |
| 3.10.22 | 1.7.2 (latest compatible) | 2.2.6 | 1.15.3 | 2.3.3 | 102 passed, 6 skipped |
| 3.11.17 | 1.9.1 | 2.4.6 | 1.17.1 | 3.1.1 | 102 passed, 6 skipped |
| 3.13.5 | 1.9.1 | 2.5.4 | 1.18.1 | 3.1.1 | 102 passed, 6 skipped |

Those six skips concern existing torch/quantum examples: these isolated ML
environments intentionally lack those extras. All requested sklearn model
families are covered, including polynomial/scaled/mixed-column pipelines,
unfitted estimators, no-probability SVC, multiclass and sparse inputs.

Testing an earlier candidate minimum, sklearn 1.3.2, exposed its F1 behavior
with `zero_division=np.nan`: an entirely wrong two-class prediction gave NaN
F1 instead of zero. Version 1.4.2 and current stable produce the correct zero.
The extra and runtime guard therefore require 1.4.2. SeeDL does not substitute
its own metric formula to mask that upstream difference.

Final checks on Python 3.12.14 / sklearn 1.9.1:

| Check | Actual result |
| --- | --- |
| Full pytest, including installed-browser tests | **919 passed, 6 skipped, 2 warnings** in 152.03 s |
| Ruff lint / format | Passed; 179 Python files formatted |
| mypy | Passed on Python 3.12 and 3.10; 63 source files |
| Lockfile and release metadata | `uv lock --check` and `verify_release.py --check-dist` passed |
| Isolated wheel + sdist build | Both built successfully; Twine checks passed |
| Fresh core-only Python 3.10 wheel install | `seedl --help`, `seedl help evaluate`, `seedl doctor`, core SVG and actionable missing-extra error passed; scientific frameworks absent from import state |
| Fresh Python 3.13 wheel with `[sklearn]` | Fitted Iris inspection/evaluation, sklearn metric comparison, offline HTML/SVG passed without pandas, torch, Qiskit or Plotly installed |
| Fresh core-only Python 3.10 sdist install | Built/installed and `python -m seedl --help` passed |
| Executable notebooks | Breast Cancer: 6 code cells; California Housing: 9 code cells; both completed without execution errors |

The six full-suite skips require CUDA/ROCm/MPS hardware unavailable on this
machine. The two warnings originate in the environment's anyio pytest plugin
(pytest deprecation) and PyTorch's existing sparse invariant check warning.
No existing tests were removed. The established CPU tensor, neural-network and
quantum checks all continue to pass. GitHub CI adds a minimum-sklearn job to
the existing Python 3.10–3.13 matrix; remote CI results are separate from these
local checks.

## Safety boundaries and remaining limitations

- Inspection reads fitted attributes and calls no prediction, transformation or
  fit. Evaluation predicts once for labels; valid score curves may additionally
  call `predict_proba` or `decision_function`. Export and plot selection never
  rerun a model. Standard estimator state and input arrays are checked for
  mutation, including `StandardScaler(copy=False)` pipelines.
- Permutation importance and brute PDP/ICE are explicit repeated-response
  diagnostics. Only `learning_curve` / `validation_curve` with
  `allow_refit=True` fit sklearn clones; they never fit the supplied estimator.
- Supervised evaluation currently supports one target per sample. Multiclass
  precision/recall/F1 and confusion matrices work; multiclass ROC/PR are
  explicitly unavailable. Sample weights and multilabel/multioutput metrics
  are not exposed in this adapter.
- New ML evaluation accepts CPU NumPy/pandas/SciPy inputs. It rejects GPU array
  protocols before copying. Existing CPU/CUDA/ROCm PyTorch paths are retained;
  no CUDA/ROCm hardware was available for new execution checks. Native Windows
  and macOS were not executed; existing plain/encoding fallback CLI tests run
  on Linux.
- DBSCAN needs caller-supplied labels and `split="training"`; matching lengths
  cannot prove row identity. KMeans prediction can evaluate new rows. A display
  PCA must already be fitted and never defines silhouette distances.
- No full sparse feature matrix is densified. Only bounded two-coordinate
  previews and sklearn's bounded silhouette distance matrices may be dense.
  Dense input/stored sparse cells are capped at 10 million, with row/feature,
  transformed-space and chart limits. Known large expansions are rejected
  before transformation. Arbitrary third-party transformers can allocate
  internally beyond what their fitted metadata predicts.
- Permutation: <=2,000 sampled rows, <=200 input features, <=20 repeats.
  PDP/ICE: <=512 rows, <=50 grid values, <=50 ICE traces, <=4 numeric features.
  Silhouette: <=1,000 sampled non-noise rows. Explicit fitting curves:
  <=5,000 rows, 2–5 folds and <=6 settings. These are guardrails, not hard
  execution timeouts for user-defined estimators/scorers.
- Expanded feature names are used only when reliably obtainable. Unknown
  mappings remain explicitly transformed coordinates. Importance and
  coefficients are associations/model responses, not causal explanations.
- User-defined prediction/scoring methods execute user code; SeeDL cannot
  guarantee they are pure. The adapter never calls `fit()` implicitly and
  protects caller data, but does not claim to sandbox arbitrary estimators.

Existing public signatures and scientific algorithms remain intact. New APIs
are additive, `Evaluation` extends `Visualization`, and `sd.see` recognizes
estimators before the established tensor/model routes. The main behavior change
is that sklearn estimators now produce an inspection instead of an unsupported
object error. No release, version bump or merge is part of this work.
