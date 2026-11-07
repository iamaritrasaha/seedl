# Developer experience audit

Inspected `main` at `01ff054` before implementation, 2026-11-07.
Python 3.12.14, CPU PyTorch 2.15.0, NumPy 2.3.5, Plotly 7.1.0,
Matplotlib 3.10.8, Qiskit 2.6.2. Baseline: **767 passed, 30 skipped,
2 warnings** (50.14 s); Ruff and mypy passed. Browser installation completed
after collection; browser tests must be rerun. Accelerator checks require hardware.

| Observed problem | Files | Proposed change | Compatibility risk |
| --- | --- | --- | --- |
| No console script or module entry point | `pyproject.toml`, package | Standard-library argparse CLI; discovery, doctor, explicit demo export | Additive; help must keep optional frameworks unloaded |
| 36 public operations are exported but there is no task-oriented discovery | `api.py`, `__init__.py` | Small shared discovery catalog, `guide()`, executable examples, live signatures/docstrings | Preserve every public function and parameter |
| Metadata is an unaligned dump; activation diagnostics appear after routine capture notes | `view.py`, `api.py` | Bounded text summary, aligned layer statistics, findings first, full-detail option | Text representation changes; underlying data and numerical engines stay intact |
| API calls print automatically in scripts; calling SVG `.show()` prints the same result again | `api.py:_show_result`, `view.py:show` | Preserve implicit terminal summaries, suppress an unchanged repeat; detect notebook kernels separately from terminal IPython | Scripts parsing exact text need updating; explicit Plotly/Matplotlib show behavior retained |
| SVG minimum width 760 px forces internal scrolling; matrix cells cap at 20 px even for 5 x 4 data | `renderers.py`, `_export.py` | Responsive SVG, larger labeled small matrices, real sequential palette and colorbar | SVG geometry/colors change, values and indices preserved |
| Generic HTML repeats title, summary and notes already embedded in SVG | `_export.py` | One document header, responsive chart, semantic text alternative with diagnostics | HTML presentation changes; offline SVG/PNG and Plotly export paths retained |
| Tensor slider row overflows: document width 486 px at 390 px viewport | `_tensor_explorer.py` | Wrapping controls, flexible grids, scoped shared typography, keyboard value inspection | Existing control IDs and interactions preserved |
| Tensor “Viridis” is a custom purple/green/red ramp, not Viridis; slice statistics use fixed four decimals (tiny nonzero values become 0.0000) | `_tensor_explorer.py` | Interpolate published Viridis anchors; significant digits/exponential display, explicit range legend | Presentation changes only; no new scientific computation |
| Different light/dark styles and hard-coded typography across renderers | `renderers.py`, `_network_renderers.py`, `_quantum_renderers.py`, specialized HTML | Shared presentation primitives on common export paths; align key tensor/gradient surfaces without rewriting engines | Keep specialized controls and user theme settings |
| README is 28 KB with release history and advanced examples before a simple install/use path | `README.md`, `docs/`, `examples/` | Short quickstart; beginner/intermediate/advanced index; retain extended reference | Fix moved relative links; no version bump |

## What worked when exercised

Real data: UCI Iris data, first five measurements, downloaded directly from
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data.
Confirmed rows against the source. Tested matrix and 3D tensor HTML/SVG exports,
network inspection and activations using those measurements, existing-gradient
health, attention probabilities (explicit synthetic example), and the analytic
quantum `+` state. All generated successfully. Screenshots taken at 390 and
1280 px before changes. Generic HTML contains its SVG in a scroll container;
this avoids page overflow but clips the chart on phones. No claim that exports
or scientific engines are generally broken.

Import already avoids optional frameworks; retain this. Existing tests cover
nonfinite handling, bounded sampling, one forward execution, model state
restoration, sparse gradients, safe serialization and offline exports. Reuse
these tests and add focused presentation/discovery regressions.

## Version verification

`pyproject.toml` and `seedl.__version__` both report 0.6.0. PyPI JSON on
2026-11-07 confirms 0.6.0 wheel and source artifacts uploaded 2026-10-29.
No version or release changes are needed. The README should distinguish
installation from PyPI and trying unreleased branch improvements from source.
