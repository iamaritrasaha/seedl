"""CNN Feature Map Explorer Example — SeeDL 0.4.0.

Demonstrates genuine activation capture, multi-channel gallery inspection,
channel diagnostics, and offline HTML export on a reproducible PyTorch CNN.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import seedl as sd


class VisionModel(nn.Module):
    """Small reproducible convolutional neural network."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        h = self.pool1(self.relu1(self.conv1(x)))
        return self.relu2(self.conv2(h))


def main():
    print("SeeDL 0.4.0 — CNN Feature Map Explorer Example")
    print("=" * 55)

    # 1. Initialize reproducible model and sample input
    torch.manual_seed(42)
    model = VisionModel()
    sample_input = torch.randn(1, 3, 28, 28)

    print("\n1. Capturing feature maps from PyTorch CNN...")
    view = sd.features(model, input=sample_input)
    print(f"   Title: {view.title}")
    print(f"   Summary: {view.summary}")
    print(f"   Captured layers: {len(view.data['layers'])}")

    # 2. Inspect layer metrics
    first_layer = view.data["layers"][0]
    first_ch = first_layer["channels"][0]
    stats = first_ch["stats"]
    print(f"\n2. Channel Diagnostics for {first_layer['name']} (Ch 0):")
    print(f"   - Min activation: {stats['min']:.4f}")
    print(f"   - Max activation: {stats['max']:.4f}")
    print(f"   - Mean activation: {stats['mean']:.4f}")
    print(f"   - Sparsity (zeros): {stats['sparsity']:.1%}")

    # 3. Compare feature maps across two distinct inputs
    print("\n3. Comparing feature maps across different inputs...")
    input_b = torch.randn(1, 3, 28, 28)
    view_b = sd.features(model, input=input_b)
    cmp_view = sd.compare(view, view_b, names=("Input A", "Input B"))
    print(f"   Comparison summary: {cmp_view.summary}")

    # 4. Export standalone offline HTML
    out_dir = Path("examples/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cnn_feature_explorer.html"
    view.save(out_file)
    print(f"\n4. Saved interactive offline HTML to: {out_file}")
    print("   Open this file in any browser to inspect channels, hover coordinates, and gallery!")


if __name__ == "__main__":
    main()
