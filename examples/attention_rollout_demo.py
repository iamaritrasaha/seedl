"""Example: Compute and visualize Attention Rollout across transformer blocks."""

from __future__ import annotations

import torch

import seedl as sd


def main():
    print("Simulating 6-layer Transformer Attention...")
    tokens = [
        "[CLS]",
        "Deep",
        "Neural",
        "Observatory",
        "Visualizes",
        "Information",
        "Flow",
        "[SEP]",
    ]
    num_layers = 6
    num_heads = 4
    seq_len = len(tokens)

    # Generate multi-layer attention matrices
    raw_attention = torch.softmax(torch.randn(num_layers, num_heads, seq_len, seq_len), dim=-1)

    print("Computing Cumulative Attention Rollout with residual blending...")
    viz_rollout = sd.attention_rollout(
        raw_attention,
        tokens=tokens,
        include_residual=True,
        head_reduction="mean",
        title="Transformer Attention Rollout Demo",
    )

    path = viz_rollout.save("transformer_rollout.html")
    print(f"Attention Rollout visualization saved to: {path}")

    print("Exploring Multi-Layer Head Entropies...")
    viz_layers = sd.attention_layers(
        raw_attention,
        tokens=tokens,
        title="Multi-Layer Transformer Attention Explorer",
    )
    print(f"Entropy progression across {num_layers} layers: {viz_layers.data['layer_entropies']}")


if __name__ == "__main__":
    main()
