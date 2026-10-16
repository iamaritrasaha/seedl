"""Comprehensive performance and memory benchmark suite for SeeDL 0.5.0."""

from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from pathlib import Path

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

import seedl as sd
from seedl._trace_renderers import trace_html


class BenchmarkModel(nn.Module):
    def __init__(self, depth: int = 10):
        super().__init__()
        layers = []
        in_c = 16
        for _ in range(depth):
            layers.append(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(in_c))
            layers.append(nn.ReLU())
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_c, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def run_benchmark(name: str, fn, budget_ms: float, budget_kb: float | None = None) -> dict:
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    result = fn()

    t1 = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    duration_ms = (t1 - t0) * 1000.0
    peak_kb = peak_mem / 1024.0

    passed_time = duration_ms <= budget_ms
    passed_mem = budget_kb is None or peak_kb <= budget_kb
    status = "PASS" if (passed_time and passed_mem) else "FAIL"

    return {
        "name": name,
        "duration_ms": duration_ms,
        "budget_ms": budget_ms,
        "peak_kb": peak_kb,
        "budget_kb": budget_kb,
        "status": status,
        "result": result,
    }


def main():
    print("=" * 70)
    print("  SeeDL 0.5.0 — Neural Observatory Benchmark Suite")
    print("=" * 70)

    if not _TORCH_AVAILABLE or not _NUMPY_AVAILABLE:
        print("PyTorch and NumPy are required to run the full benchmark suite.")
        sys.exit(1)

    benchmarks = []

    # 1. Execution Tracer (10-layer ConvNet)
    model = BenchmarkModel(depth=10)
    x = torch.randn(1, 16, 16, 16)
    benchmarks.append(
        run_benchmark(
            "sd.trace (10-layer CNN, 32 nodes)",
            lambda: sd.trace(model, input=x, max_nodes=128),
            budget_ms=150.0,
            budget_kb=5000.0,
        )
    )

    trace = benchmarks[-1]["result"]

    # 2. Tensor Flow Explorer HTML Export
    flow_viz = sd.flow(trace)
    benchmarks.append(
        run_benchmark(
            "sd.flow HTML generation",
            lambda: len(trace_html(flow_viz).encode("utf-8")),
            budget_ms=50.0,
            budget_kb=2000.0,
        )
    )

    # 3. Transformer Attention Rollout (12 layers, 12 heads, 32 tokens)
    rng = np.random.RandomState(42)
    raw = rng.uniform(0.1, 1.0, size=(12, 12, 32, 32))
    attn_matrices = raw / raw.sum(axis=-1, keepdims=True)
    benchmarks.append(
        run_benchmark(
            "sd.attention_rollout (12L x 12H x 32T)",
            lambda: sd.attention_rollout(attn_matrices),
            budget_ms=80.0,
            budget_kb=5000.0,
        )
    )

    # 4. Integrated Gradients (50 steps)
    linear_model = nn.Sequential(nn.Linear(20, 50), nn.ReLU(), nn.Linear(50, 2))
    lin_x = torch.randn(1, 20)
    benchmarks.append(
        run_benchmark(
            "sd.explain (IG, 50 steps)",
            lambda: sd.explain(linear_model, input=lin_x, method="integrated_gradients", steps=50),
            budget_ms=150.0,
            budget_kb=5000.0,
        )
    )

    # 5. SmoothGrad (25 samples)
    benchmarks.append(
        run_benchmark(
            "sd.explain (SmoothGrad, 25 samples)",
            lambda: sd.explain(linear_model, input=lin_x, method="smoothgrad", samples=25),
            budget_ms=120.0,
            budget_kb=5000.0,
        )
    )

    # 6. Gradient Health (Inspection & Deduplication)
    loss = linear_model(lin_x).sum()
    loss.backward()
    benchmarks.append(
        run_benchmark(
            "sd.gradient_health (MLP)",
            lambda: sd.gradient_health(linear_model),
            budget_ms=50.0,
            budget_kb=2000.0,
        )
    )

    # 7. Quantum Noise Simulation (Depolarizing, 20 steps)
    benchmarks.append(
        run_benchmark(
            "sd.quantum_noise (20 steps)",
            lambda: sd.quantum_noise(channel="depolarizing", param=0.5, steps=20),
            budget_ms=50.0,
            budget_kb=2000.0,
        )
    )

    # 8. Scientific Report Generation (3 sections)
    rep = sd.report(
        "Benchmark Report",
        views=[flow_viz, benchmarks[2]["result"], benchmarks[5]["result"]],
        subtitle="Performance Validation",
    )
    benchmarks.append(
        run_benchmark(
            "sd.report HTML assembly",
            lambda: len(rep.to_html().encode("utf-8")),
            budget_ms=100.0,
            budget_kb=10000.0,
        )
    )

    # 9. Analysis Serialization Roundtrip
    benchmarks.append(
        run_benchmark(
            "sd.save_analysis / load_analysis roundtrip",
            lambda: _test_roundtrip(flow_viz),
            budget_ms=50.0,
            budget_kb=2000.0,
        )
    )

    # Print results
    print(
        f"\n{'Benchmark':<45} | {'Time (ms)':<10} | {'Budget (ms)':<11} | {'Peak (KB)':<10} | {'Status'}"
    )
    print("-" * 90)
    all_passed = True
    for b in benchmarks:
        status_sym = "✔ PASS" if b["status"] == "PASS" else "✘ FAIL"
        if b["status"] != "PASS":
            all_passed = False
        print(
            f"{b['name']:<45} | {b['duration_ms']:<10.2f} | {b['budget_ms']:<11.1f} | {b['peak_kb']:<10.1f} | {status_sym}"
        )

    print("-" * 90)
    if all_passed:
        print("ALL BENCHMARKS SATISFIED PERFORMANCE BUDGETS SUCCESSFULLY.")
    else:
        print("WARNING: Some benchmarks exceeded their performance budget.")
        sys.exit(1)


def _test_roundtrip(viz):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "test.json"
        sd.save_analysis(viz, p)
        return sd.load_analysis(p)


if __name__ == "__main__":
    main()
