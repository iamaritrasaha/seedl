"""Example: Assemble a comprehensive, publication-grade scientific HTML report."""

from __future__ import annotations

import torch
import torch.nn as nn

import seedl as sd


def main():
    print("Generating components for scientific report...")

    # 1. Flow Tracer
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    x = torch.randn(1, 10)
    trace = sd.trace(model, input=x)
    flow_viz = sd.flow(trace)

    # 2. Attention Rollout
    tokens = ["[CLS]", "scientific", "ai", "observatory", "[SEP]"]
    attn = torch.softmax(torch.randn(4, 2, 5, 5), dim=-1)
    rollout_viz = sd.attention_rollout(attn, tokens=tokens)

    # 3. Quantum Noise
    quantum_viz = sd.quantum_noise(channel="depolarizing", param=0.35, steps=8)

    print("Assembling multi-section Scientific Report...")
    report = sd.report(
        title="Neural Observatory & Quantum Intelligence Comprehensive Audit",
        subtitle="Experimental Verification Sprint",
        abstract=(
            "This report documents forward tensor execution, attention propagation across "
            "transformer layers, and open quantum system decoherence using SeeDL 0.5.0."
        ),
        author="Principal AI & Quantum Research Engineer",
        affiliation="Institute for Advanced Scientific Computing",
        theme="dark",
    )

    report.add_section(
        title="Neural Execution Tracing",
        views=[flow_viz],
        narrative="Forward pass execution observed with zero parameter mutation.",
    )

    report.add_section(
        title="Transformer Attention Rollout",
        views=[rollout_viz],
        narrative="Cumulative token-to-token information routing with row stochasticity.",
    )

    report.add_section(
        title="Open Quantum System Decoherence",
        views=[quantum_viz],
        narrative="Kraus depolarizing channel simulation satisfying CPTP completeness.",
    )

    report_path = report.save("comprehensive_observatory_report.html")
    print(f"Comprehensive Scientific Report successfully saved to: {report_path}")


if __name__ == "__main__":
    main()
