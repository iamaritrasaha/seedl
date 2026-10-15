"""Real MLP training, weight/gradient playback, comparison, and quantum evolution.

Run: uv run --extra all python examples/live_neural_intelligence.py --output-dir /tmp/seedl-live
In a notebook, leave ``run.playback()`` or ``sd.circuit(qc, simulate=True)`` as the last expression.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from qiskit import QuantumCircuit
from torch import nn

import seedl as sd


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
    x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    target = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    with sd.watch(model, name="XOR learning", max_snapshots=24, max_bytes=2 * 1024 * 1024) as run:
        # Record the initial and trained states using the same input batch.
        for epoch in range(21):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = (prediction - target).square().mean()
            loss.backward()
            run.capture(step=epoch, epoch=epoch, loss=loss)
            run.log(epoch=epoch, loss=loss.item())
            if epoch < 20:
                optimizer.step()

    # Connection weights, activations, loss and step navigation share one view.
    run.playback().save(output_dir / "training.html")
    # Existing gradients are read; no second backward pass is run by SeeDL.
    run.playback(color_by="gradient").save(output_dir / "gradients.html")
    # Raw shapes, flat indices, weights, gradients and outputs remain inspectable.
    initial, trained = run.snapshots[0], run.snapshots[-1]
    sd.compare(initial, trained, names=("Initial", "Trained")).save(output_dir / "comparison.html")

    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.7, 1)
    qc.measure(0, 0)
    qc.x(1)  # Playback marks post-measurement states unavailable.
    sd.circuit(qc, simulate=True).save(output_dir / "quantum.html")
    print(f"Saved playback HTML to {output_dir.resolve()}")
    print(f"Retained recording: {run.recording_stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("live-neural-output"))
    main(parser.parse_args().output_dir)
