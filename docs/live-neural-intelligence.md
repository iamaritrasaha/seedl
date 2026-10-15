# Live neural intelligence — 0.3.0 development (unreleased)

The package version remains **0.2.0**. This branch adds recorded training playback and gate-by-gate quantum inspection. Run the complete [example](../examples/live_neural_intelligence.py) with:

```bash
uv run --extra all python examples/live_neural_intelligence.py --output-dir /tmp/seedl-live
```

## Record actual training passes

```python
import seedl as sd
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.tensor([[1.0, -1.0]])

with sd.watch(model, name="MLP", max_snapshots=32, every=1) as run:
    for epoch in range(20):
        optimizer.zero_grad(set_to_none=True)
        output = model(x)  # Your real forward pass
        loss = (output - 0.5).square().mean()
        loss.backward()  # Your real backward pass
        run.capture(step=epoch, epoch=epoch, loss=loss)
        optimizer.step()  # Capture MUST precede this

run.playback()  # Inline notebook controls
# sd.see(run) is equivalent when snapshots are available.
run.playback().save("training.html")  # Standalone, offline, no kernel
run.playback(color_by="gradient").save("gradients.html")
```

`sd.track(model, ...)` is an alias of `sd.watch(model, ...)`. `Tracker(name="run", model=model, ...)` also accepts capture options. Recording starts on context entry and hooks are removed on exit, including exceptions. SeeDL observes the user's actual pass; capture never evaluates the model, runs backward, updates parameters, changes training mode, or consumes random numbers. Forward/backward remain entirely the caller's responsibility.

Call `capture()` once per completed forward pass. Gradients are those present at capture, so clear stale gradients before the pass and call backward first when gradients are wanted. A forward without backward is valid and records missing gradients as unavailable. Gradient accumulation is supported: the snapshot then contains the accumulated gradients. Loss is an optional scalar supplied by the caller. Keep the same input batch across recordings when interpreting output changes as a parameter effect.

A standard parameter update between forward and capture is rejected using parameter identity, version and shape. Capture cannot detect unsupported direct `.data` mutation, concurrent model changes, or prove that a caller-supplied loss/gradient belongs to a particular pass. Use a single training thread; custom hooks/forward implementations remain user code, not a sandbox. Observation hooks do not replace inputs or outputs. Failed forwards cannot be captured.

## Bounds and provenance

| Option | Default | Meaning |
| --- | ---: | --- |
| `max_snapshots` | 64 | Maximum retained snapshots; oldest evicted |
| `max_bytes` | 8 MiB | Sum of encoded retained snapshot bytes; oldest evicted |
| `every` | 1 | Record every nth capture call, starting with the first |
| `max_values` | 2048 | Global parameter-value sample budget, and separate output sample budget |
| `max_neurons` | 100 | Exact neuron graph budget; also capped at 48 per layer / 12 layers |
| `max_connections` | 1500 | Deterministic connection sample budget |
| `sample` | 0 | Zero-based batch row used for exact neuron activations |

A snapshot larger than the byte budget raises an error. Parameter capture supports up to 256 initialized dense real parameter tensors. It gathers indexed values before copying to CPU, including noncontiguous tensors. The store contains immutable JSON bytes, with no tensors or autograd graphs. `snapshot.data` decodes an independent copy. Python container overhead, the bounded pending forward preview, decoded data, user-held snapshots, figures, and HTML construction are outside the encoded byte budget. `run.recording_stats` reports retained bytes, count and evictions. Legacy `log()/history()/save()/load()` are unchanged and metric history is independently unbounded; tracker JSON persistence saves metrics, not training recordings.

Neuron wiring is available for the existing supported exact `Linear` / nested `Sequential` MLPs and activations. Nodes have stable `layer:index` identities. Edges use **`weight[destination, source]`**, and node gradient colors represent existing **bias gradients**; a bias-free neuron has no invented bias gradient. Edge hovers contain weight gradients. The status shows the maximum absolute gradient in the retained parameter sample. Weight widths and activation/gradient color scales are fixed across retained steps so playback values can be compared.

Other models and graphs above the neuron budget use indexed parameter/output previews. They do not infer neuron connections or fabricate intermediate activations. Non-finite values and absent gradients remain unavailable. No optimizer steps or interpolation are synthesized. Training snapshot step labels must strictly increase; epoch labels are optional finite numbers. Omitted step labels advance from the last explicit or automatic step. Timeline positions refer to retained snapshots, so evictions and `every` can produce gaps in the original step sequence.

## Playback and comparison

The HTML controls provide play, pause, rewind, previous/next, speed, recorded-step selection, epoch selection, and a timeline. A loss curve follows the active recorded step. The panel reports retained snapshot bytes, eviction count, and per-step parameter/output/connection coverage (or displayed quantum basis coverage). Select or click a neuron to inspect its current values; selection and the 3D camera persist during updates. Plotly is required for playback; SVG and Matplotlib show the first retained observation. `Visualization.figure` / `to_figure("plotly")` expose the first frame; inline notebook display and `.save("file.html")` / `.to_html()` provide the full controls. A composite export embeds one Plotly runtime, and multiple inline outputs reuse an already-loaded runtime.

```python
initial, trained = run.snapshots[0], run.snapshots[-1]
comparison = sd.compare(initial, trained, names=("Initial", "Trained"))
comparison.save("comparison.html")
raw = comparison.data
```

Snapshot comparisons include sampled weight changes, activation distributions, gradient statistics, and output changes. Weight/output deltas align parameter names, tensor shapes and original flat indices. Shape mismatches, missing parameters and non-overlapping samples are marked explicitly. Statistics describe retained samples, not full tensors or a dataset-wide activation distribution. The original `sd.compare(model_a, model_b)` architecture comparison remains supported.

## Quantum gate inspection

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 1)
qc.h(0)
qc.cx(0, 1)
qc.rz(0.7, 1)
qc.measure(0, 0)
qc.x(1)
view = sd.circuit(qc, simulate=True, max_qubits=10, max_ops=500, max_bytes=32 * 1024 * 1024)
view.save("quantum.html")
view  # Notebook: operation selector, timeline, before/after panes
```

`sd.see(qc, simulate=True)` uses the same interactive playback. `simulate=False` retains the circuit diagram. Select an operation from the dropdown or click a gate in the preserved circuit diagram. The selected operation shows the valid before and after reduced-qubit Bloch vectors, computational-basis probabilities and complex amplitude phases. Hover the probability bars for real/imaginary amplitudes and phase in radians. Basis labels use Qiskit's ordering (qubit 0 is the least significant bit). For more than 64 basis states, charts show the first 64; full bounded data remains available in `view.data["evolution"]`. A zero amplitude has no physically defined phase; its stored phase follows `cmath.phase(0) == 0`.

Measurement, reset, conditional operations and unsupported/unbound gates halt pure-state evolution. At the first halt, only the valid **before** state appears; the after pane and all subsequent states are unavailable. Barriers never resume halted evolution. Legacy evolution entries retain the last pre-halt values for compatibility, explicitly labeled `statevector_valid=False` and `state_role="last valid pre-halt state"`; they are never presented as valid post-operation states. Entangled qubits show reduced Bloch vectors inside the sphere, not invented pure states. Mathematical regression checks compare amplitudes, probabilities, phases and each reduced qubit's Pauli expectations with `qiskit.quantum_info` after every supported gate.

## Limits

Serialized HTML content (frames, controls, initial figure and circuit diagram) is capped at 32 MiB before embedding the Plotly runtime. Reduce retained snapshots, neuron/connection budgets or circuit size if export exceeds the cap. Export construction needs temporary CPU memory proportional to the frame payload; the recorder's byte budget is not a process-wide memory limit. Playback is recorded observation, not a background training dashboard. New CNN channel-selection and feature-map comparison controls are deferred; existing CNN visualization remains available.

## Sprint verification — 2026-10-15

- Full Python 3.13 suite: **330 passed, 1 skipped** (CUDA/ROCm unavailable). An additional browser regression for literal template-like labels and script-tag escaping passed after the full run.
- `ruff check .`, `ruff format --check .`, `mypy`, `uv lock --check`, `uv build --no-sources`, and `git diff --check` passed. Package metadata remains 0.2.0.
- Python 3.10, 3.11, 3.12 and 3.13 passed dependency-free import, SVG, metric-tracker and source-syntax smoke checks. Full optional-dependency testing was performed on 3.13 only.
- Playwright Chromium tested actual WebGL neuron clicks, actual SVG gate clicks, timeline and epoch selection, play/pause/rewind/speed, recorded values, camera preservation, offline export, independent inline views and safe label serialization. Rendered screenshots were inspected.
- A fresh Jupyter kernel executed five cells with real PyTorch training and Qiskit simulation, producing three interactive HTML outputs. The notebook's exported HTML rendered and operated in Chromium with HTTP blocked. This verifies kernel execution and exported notebook outputs; a live JupyterLab frontend was not tested.
- Quantum regressions compared amplitudes, probabilities, phases and reduced Pauli expectations with `qiskit.quantum_info` after every gate for 1–4 qubits. Measurement, reset, conditional/unbound operations and barriers after halt were tested.
- A CPU 4→8→2 MLP benchmark recorded 1,000 passes: median capture **0.837 ms** (first 100 passes, tracing disabled). With a **128 KiB** budget and eight-snapshot cap, the store retained **119,135 bytes / seven snapshots** and evicted 993. Python allocation tracing over the remaining passes measured a 444,219-byte peak; it excludes native tensor/GPU allocations. Export was 5,204,409 bytes including Plotly. These are machine- and workload-specific measurements, not process-wide memory bounds. GPU memory behavior was not exercised.

New CNN channel-selection/feature-map comparison controls remain deferred. Existing CNN, static neuron/SVG, metric tracking and quantum behavior are covered by the full suite.
