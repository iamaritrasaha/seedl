# Performance Engineering, Benchmarks & Memory Budgets

## 1. Performance Philosophy

Interactive visualization tools must be fast and responsive. A developer should never wait tens of seconds for an activation tracer or attention inspector to render, nor should an in-browser visualization freeze a machine with unbounded memory allocations.

SeeDL 0.5.0 is designed around **strict latency and memory budgets**:

1. **Sub-100ms Tracing**: Dynamic execution capture of standard 10–50 layer deep networks finishes in under 50 milliseconds on modern CPUs.
2. **Vectorized NumPy / PyTorch Acceleration**: Heavy matrix multiplications (e.g. attention rollout) utilize vectorized operations where available, running in sub-40 milliseconds across 12+ transformer layers.
3. **Bounded Sampling Limits**: Raw tensor distributions are sampled up to bounded limits ($N \le 2048$) to ensure browser rendering never encounters DOM explosion or megabyte-sized HTML payloads.
4. **Zero Pickle Footprint**: Artifact serialization utilizes pure JSON with streaming SHA-256 integrity checks.

---

## 2. Benchmark Suite Results

Benchmarks executed via `scripts/benchmark_observatory.py` on an Intel/AMD Linux workstation (Python 3.13):

| Benchmark Scenario | Observed Latency | Latency Budget | Peak Memory | Memory Budget | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`sd.trace` (10-Layer CNN, 34 nodes)** | 38.2 ms | 150.0 ms | 573 KB | 5.0 MB | **✔ PASS** |
| **`sd.flow` HTML Rendering** | 5.5 ms | 50.0 ms | 334 KB | 2.0 MB | **✔ PASS** |
| **`sd.attention_rollout` (12L x 12H x 32T)** | 36.7 ms | 80.0 ms | 1.3 MB | 5.0 MB | **✔ PASS** |
| **`sd.explain` (Integrated Gradients, 50 steps)**| 17.2 ms | 150.0 ms | 64 KB | 5.0 MB | **✔ PASS** |
| **`sd.explain` (SmoothGrad, 25 samples)** | 8.9 ms | 120.0 ms | 13 KB | 5.0 MB | **✔ PASS** |
| **`sd.gradient_health` (MLP Parameter Audit)** | 2.2 ms | 50.0 ms | 57 KB | 2.0 MB | **✔ PASS** |
| **`sd.quantum_noise` (Kraus simulation, 20 steps)**| 8.8 ms | 50.0 ms | 173 KB | 2.0 MB | **✔ PASS** |
| **`sd.report` HTML Assembly (3 Multi-Views)** | 6.3 ms | 100.0 ms | 251 KB | 10.0 MB | **✔ PASS** |
| **`sd.save_analysis` / `load_analysis` Roundtrip** | 16.0 ms | 50.0 ms | 395 KB | 2.0 MB | **✔ PASS** |

---

## 3. How to Run the Benchmarks Locally

```bash
uv run python scripts/benchmark_observatory.py
```
