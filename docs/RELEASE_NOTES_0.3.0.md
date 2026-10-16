# SeeDL 0.3.0: Live Neural Intelligence

**Status:** release candidate source prepared for review; not yet published to PyPI. Version metadata is 0.3.0. Publication, tagging and deployment remain separate manual actions.

## Training recording and interactive playback

- Record the user's real PyTorch forward passes, connection weights, selected activations, existing gradients, scalar loss and outputs across training steps.
- Navigate recorded steps and epochs with play, pause, rewind, previous/next and speed controls. A loss curve follows the active step.
- Inspect supported MLPs in the existing Plotly/WebGL 3D renderer, with stable neuron identities, fixed weight/color scales, neuron inspection and camera preservation.
- Compare initial and trained snapshots: sampled weight changes, activation distributions, gradient statistics and output changes. Parameter deltas align names, shapes and original flat indices; connections use `weight[destination, source]`.
- Export standalone HTML with offline controls and no running kernel. Exported views report retained bytes, evictions and sampling coverage. Inline notebook outputs reuse a loaded Plotly runtime. SVG and Matplotlib remain static fallbacks.

No intermediate training states are invented. Capture performs no extra forward pass, backward pass or optimizer update, and does not change model parameters, training modes, buffers or random state.

## Public APIs

```python
with sd.watch(model, name="MLP", max_snapshots=64, max_bytes=8 * 1024 * 1024) as run:
    # In your training loop: zero gradients, forward, calculate loss, backward.
    run.capture(step=step, epoch=epoch, loss=loss)  # Before optimizer.step()
    optimizer.step()

run.playback().save("training.html")
run.playback(color_by="gradient").save("gradients.html")
sd.compare(run.snapshots[0], run.snapshots[-1]).save("comparison.html")
```

- `sd.watch(model, ...)` and `sd.track(model, ...)` create bounded training recorders when used as context managers. Metric-only usage remains supported.
- `Tracker.capture(step=None, epoch=None, loss=None)` captures a completed observed pass. Omitted steps advance from the last explicit or automatic label; labels must strictly increase.
- `Tracker.snapshots` exposes immutable `TrainingSnapshot` objects. `snapshot.data` returns an independent decoded copy; `snapshot.nbytes` reports encoded size.
- `Tracker.recording_stats` reports retained snapshot count, bytes and evictions.
- `Tracker.playback(backend="auto", color_by="activation")` and `sd.see(tracker)` render recorded snapshots; `color_by="gradient"` inspects existing gradients.
- `sd.compare(snapshot_a, snapshot_b)` adds aligned recorded-state comparison. Existing model, history and prediction comparisons remain available.

See the [full API and bounds](live-neural-intelligence.md) and [runnable MLP/quantum example](../examples/live_neural_intelligence.py).

## Quantum gate evolution

`sd.circuit(qc, simulate=True)` and `sd.see(qc, simulate=True)` provide a recorded gate timeline. Select an operation or click a gate in the circuit diagram to inspect valid before/after reduced-qubit Bloch vectors, computational-basis probabilities and phase-colored complex amplitudes. Basis ordering follows Qiskit, with qubit 0 as the least significant bit.

Measurement, reset, conditional and unsupported/unbound operations halt pure-state evolution. Only the valid before state appears at the first halt; the after state and subsequent states remain unavailable. Barriers do not resume halted evolution. Legacy data retains explicitly labeled pre-halt observations, never presented as valid post-operation states. Mathematical regressions compare amplitudes, probabilities, phases and reduced Pauli expectations with `qiskit.quantum_info`.

`sd.circuit()` exposes `max_qubits=10`, `max_ops=500` and `max_bytes=32 * 1024 * 1024` simulation limits. Non-simulated circuit diagrams and the existing quantum APIs remain available.

## Performance, memory and scientific limits

- Recording defaults to 64 snapshots and 8 MiB of retained encoded payloads. Oldest snapshots are evicted to satisfy both limits; a single oversized snapshot is rejected. `every` controls capture-call sampling.
- The store contains CPU JSON bytes, with no retained tensors or autograd graphs. Parameter/output previews and connection counts are bounded; sampling and evictions are reported explicitly. Non-finite or absent values stay unavailable.
- The byte budget excludes Python object overhead, the bounded pending preview, temporary rendering allocations, metric history and snapshots retained separately by callers. It is not a process-wide or GPU memory limit.
- Parameter previews default to 2,048 values globally; outputs have a separate 2,048-value budget. At most 256 initialized, dense, real parameter tensors are supported. Sparse gradients and unsupported structured outputs are unavailable.
- Exact neuron wiring covers supported built-in Linear/Sequential MLPs, within neuron/layer limits. Larger and other models use indexed parameter/output previews. Node gradient coloring represents bias gradients; bias-free neurons have no invented bias gradient.
- Serialized HTML content is capped at 32 MiB before embedding the fixed Plotly runtime. Large recordings may require lower capture limits. Building exports needs temporary CPU memory proportional to the content.
- Clear stale gradients before training passes. Gradients and optional loss are those supplied/present at capture; accumulation is preserved. Standard parameter updates between forward and capture are rejected. Direct `.data` mutation, concurrent changes and arbitrary custom forward/hook code are outside this guarantee.
- Compare the same inputs when attributing output changes to parameter changes. Activation and gradient statistics describe retained samples, not complete datasets.
- Quantum charts show at most the first 64 basis states; full bounded state data remains inspectable. Zero-amplitude phases follow the stored mathematical convention and have no physical significance.
- GPU memory behavior and a live JupyterLab frontend are unverified in the local release environment. Kernel execution and browser-rendered notebook exports are checked separately. New CNN channel-selection and feature-map comparison controls are deferred; existing CNN functionality remains supported.

## Compatibility and publication

Python requires 3.10 or newer. The core still has **zero mandatory dependencies**; PyTorch, NumPy, Plotly, Matplotlib and Qiskit remain optional extras. Existing metric tracker JSON persistence remains metric-only. `Visualization.figure` exposes the first playback frame; inline notebook display, `.to_html()` and `.save("file.html")` provide the complete controls.

See [0.2.0 release notes](RELEASE_NOTES_0.2.0.md) for the preceding release. Older release records and existing demonstrations are preserved. This document does not establish that 0.3.0 has been published.


## Local release verification

- Full Python 3.13 suite: **337 passed, 1 skipped** (CUDA/ROCm unavailable to the installed PyTorch build). No full suite was repeated during release preparation.
- All 55 targeted recording, quantum and release-verifier checks passed before the full run. The suite covers actual recorded values, model preservation, absent autograd retention, snapshot/HTML bounds, eviction reporting, parameter indexing, optimizer-step semantics, safe serialization and measurement/reset halting.
- Chromium tests passed for offline playback, actual WebGL neuron and SVG gate clicks, camera/identity preservation, timeline/speed controls and export coverage. Five fresh notebook cells executed and produced three interactive outputs; their exported HTML operated with HTTP blocked. JupyterLab is not installed locally, so this is not a live JupyterLab frontend result.
- Clean wheel installations passed core import, version, SVG, tracker, missing-optional-dependency and source-syntax checks on Python **3.10, 3.11, 3.12 and 3.13**. A clean sdist installation passed the same checks on **3.13**. Full optional-dependency testing was local to 3.13; the merged implementation's CI previously passed the 3.10–3.13 matrix.
- Ruff lint/format, mypy, `uv lock --check`, distribution build, `scripts/verify_release.py --tag v0.3.0 --check-dist` and whitespace checks passed. Wheel metadata confirmed zero mandatory dependencies. The tag argument validates metadata only; no release tag was created.

The release remains unpushed and unpublished. GPU memory verification and a live JupyterLab smoke test remain limitations, not passing checks.
