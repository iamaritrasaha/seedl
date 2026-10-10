# SeeDL 0.1.0 release review — 10 October 2026

Recommendation: ready for a first public MVP release on the tested CPU/Linux paths, with the documented limitations. No publishing, pushing, tagging, or GitHub release was performed. The checkout was clean at the start; existing committed implementation was retained.

## Usability audit

All nine kinds were inspected in the adapters and all three renderers, exercised in a real notebook kernel, and reviewed visually in SVG and Matplotlib outputs. Plotly frontend checks included interactive zoom, RGB colors, and activation distributions; not every interactive control was exercised.

| View | Findings and focused changes |
| --- | --- |
| Scalar | Retained the simple value view and explicit non-finite text. Added visible metadata consistently. |
| Sequence | Sampled vectors falsely renumbered their points. All backends now retain zero-based source indices; SVG has numeric x-axis endpoints and overflow-safe coordinate normalization. |
| Matrix | SVG lacked a value scale; downsampled coordinates were lost. Added source row/column indices, a scale explanation and cell hover values. Matrix previews now respect `max_points`. |
| Image | Plotly treated normalized RGB values using 255-based defaults; one-channel images needed an appropriate grayscale representation. Fixed component ranges and RGBA alpha. NumPy and Torch scaling now agree for supported integer/float images, including extreme finite values. Preview pixel counts are bounded and assumptions are stated. |
| Distribution | SVG histograms lacked numeric value/count context. Added range/count labels and an explicit empty/non-finite state. Finite values are filtered before optional-backend histograms. Histograms retain backend-specific binning. |
| Model | Dense views had inconsistent row limits and long SVG labels overlapped columns. All tables now show the first 100 rows, disclose omissions, and retain full data. SVG shortening preserves full hover text; Matplotlib tables use readable fixed text sizes and bounded figure height. Clarified direct parameter counts, shared parameters, and the difference between a hierarchy and an execution graph. |
| Training | Validation series could have indistinguishable names. Preserve original metric names, distinguish validation with dashed lines, sample large series with original epochs and disclosure, reject non-finite epochs, and suppress overfitting advice when the curves' epoch coordinates do not match. |
| Activations | Plotly distributions omitted shapes and finite summaries. Added a bounded summary table. Numeric module names could merge/misplace box distributions: explicit category axes now keep Sequential layer distributions separate, including nonnegative ReLU samples. Charts show the first 20 modules; full summaries remain available. Empty outputs no longer cause sampling division errors. |
| Gradients | Missing-gradient status was hidden by shape text in SVG, and omitted from optional-backend charts. All backends now expose shapes, status, finite state and summaries. Plots show the first 20 parameters, tables the first 100. Numeric names use categorical axes. |

SVG titles/descriptions and wrapped notes expose shape, dtype, device, statistics, gradient tracking, and non-finite status. Narrow outputs retain a minimum readable width and may require horizontal scrolling. Plotly and Matplotlib share core summaries, coordinates and status information; their supplementary plots and histogram bins need not be identical.

## Notebook evidence

The isolated Python 3.12 environment contained CPU Torch 2.14.1, NumPy 2.5.3, Plotly 7.1.0, Matplotlib 3.11.2 and JupyterLab 4.6.4. The notebook contains 29 code cells: setup; nine kinds in each of three backends; and unusual-input/dense-model checks. It was executed with nbclient and then through JupyterLab's Restart and Run All. Final outputs were saved and checked for execution errors and stderr.

A real frontend defect was found: `_repr_html_` supplied SVG for Matplotlib views, and JupyterLab preferred that HTML over the generated PNG. It now supplies HTML only for SVG. Actual PNG images were subsequently verified in JupyterLab. Plotly zoom visibly changed the sequence's axis range. Plotly HTML and Plotly JSON output were both exercised; the HTML notebook renderer was explicitly selected alongside the JSON renderer. Saved HTML was also visually inspected in a browser.

Evidence is under `dist/review/`: the executed `notebook-checks.ipynb`, exported `notebook-checks.html`, PNG outputs, and browser screenshots including `plotly-zoom.jpg`, `jupyter-matplotlib-png.jpg` and `jupyter-activations-final.jpg`. The screenshot named
`plotly-activations-before-category-fix.jpg` records the defect before correction. These review artifacts are excluded from release packages. Other notebook frontends and every individual Plotly interaction were not verified.

## Reliability and model execution

- Standard-layer tests verified tensor inputs, parameters, existing gradients, mixed per-module training flags and registered buffer values. Error-path tests checked temporary hook cleanup and flag restoration.
- Tensor inputs are detached/cloned. Registered buffers are copied/restored, including ordinary replacement/addition, resized buffers and non-persistent-buffer registration metadata. An additional custom `.data` dtype-mutation probe demonstrated that direct storage manipulation can defeat value restoration; this boundary is now explicitly documented. Non-tensor objects and arbitrary custom model behavior are not sandboxed.
- An explicit regression test demonstrates that a custom forward method's Python counter and parameter mutation persist. This is documented; arbitrary execution is not described as side-effect-free. Random state, files, custom Python attributes and custom gradient/parameter mutations are not restored.
- Invalid backend requests now fail before running the model. Plain numeric SVG calls no longer unnecessarily import Torch or NumPy.
- Full-array statistic chunks are bounded even for high-rank non-contiguous arrays. NumPy moments use scaled arithmetic to avoid finite-value overflow/underflow. Tests cover independently known extreme-value statistics and chunk limits.
- Forward-pass memory still includes full input/buffer copies and normal model allocations. History/Python sequence validation can make full-size copies. No peak-RSS, GPU-memory or arbitrary-model runtime bound was measured or promised.

## Validation and packaging

- Current isolated CPU environment: 50 passed, 1 skipped (accelerator unavailable).
- Python 3.10.22 with installed wheel and older compatible dependencies: Torch 2.0.1+cpu, NumPy 1.23.5, Plotly 5.18.0, Matplotlib 3.7.0. 50 passed, 1 skipped against the installed wheel; the older Matplotlib stack emits upstream Pyparsing deprecation warnings.
- Ruff, mypy, `uv lock --check`, `git diff --check`, and `uv build` were run. All README Python blocks were executed in order, supplying a tensor for the opening shorthand example. The gradient example now creates a runnable scalar objective.
- Built wheel and source distribution were inspected with archive and metadata checks. Version remains 0.1.0 in API and distribution metadata; MIT license text and author attribution are consistent; core has no mandatory third-party dependencies; all six extras remain correctly conditional.
- Added `py.typed` and repository/issue metadata links. Source distributions now include tests and `uv.lock`; wheels exclude tests, environments, caches, and review outputs.
- Clean core-wheel and source-distribution installs exercised standard-library SVG behavior and actionable missing-extra errors. The older compatible environment ran the full suite against the installed wheel. Installed dependencies were checked with `uv pip check`.

## Remaining release limitations

CPU/Linux and JupyterLab are the verified platform/frontend combination. GPU execution, Windows/macOS, Colab, VS Code notebooks and other frontends remain unverified. Registered-buffer restoration does not guarantee recovery from custom direct storage/dtype manipulation. Model observation is one eval/no-grad pass, shows the last observed call for reused modules, and does not capture keyword input shapes. Samples can miss spikes or leave layers without samples at very small budgets. Statistics are unavailable when any non-finite values occur; sparse summaries omit implicit zeros. Floating-point reductions can round integers above 2**53. Complex and quantized tensors are intentionally unsupported; lazy models must be initialized by the caller. Advanced tracing, attention/filter views and exported reports are outside this release.

Renderer and build reference checks used the official [Plotly image component documentation](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Image.html) and [uv build file-inclusion documentation](https://docs.astral.sh/uv/concepts/build-backend/#file-inclusion-and-exclusion).
