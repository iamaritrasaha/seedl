"""Transformer Attention Explorer Example — SeeDL 0.4.0.

Demonstrates genuine multi-head self-attention, cross-attention,
token-to-token alignment, head comparisons, and offline HTML export.
"""

from __future__ import annotations

from pathlib import Path

import torch

import seedl as sd


def main():
    print("SeeDL 0.4.0 — Transformer Attention Explorer Example")
    print("=" * 55)

    torch.manual_seed(42)

    # 1. Multi-head self-attention over a sentence
    sentence = ["Attention", "is", "all", "you", "need", "today"]
    num_heads = 4
    seq_len = len(sentence)

    # Generate genuine softmax attention distributions
    logits = torch.randn(num_heads, seq_len, seq_len)
    attention_weights = torch.softmax(logits, dim=-1)

    print("\n1. Visualizing Multi-Head Self-Attention...")
    view = sd.attention(attention_weights, tokens=sentence)
    print(f"   Summary: {view.summary}")
    print(f"   Mean entropy: {view.data['mean_entropy']:.2f} bits")
    print(f"   Heads captured: {view.data['captured_heads']}")

    # 2. Compare Head 0 vs Head 1
    print("\n2. Comparing Head 0 vs Head 1...")
    cmp_view = sd.attention(attention_weights, tokens=sentence, compare_heads=(0, 1))
    cmp_stats = cmp_view.data.get("head_comparison", {})
    if cmp_stats.get("available"):
        print(f"   MAE between Head 0 and Head 1: {cmp_stats['mae']:.4f}")
        print(f"   Cosine similarity: {cmp_stats['cosine_similarity']:.4f}")

    # 3. Cross-Attention with unequal query and key dimensions
    print("\n3. Visualizing Cross-Attention (Decoder attending to Encoder)...")
    dec_tokens = ["Summary", "Output"]
    enc_tokens = ["Attention", "is", "all", "you", "need", "today"]
    cross_weights = torch.softmax(torch.randn(num_heads, len(dec_tokens), len(enc_tokens)), dim=-1)

    cross_view = sd.attention(cross_weights, query_tokens=dec_tokens, key_tokens=enc_tokens)
    print(
        f"   Cross-attention shape: {cross_view.data['query_len']} queries × {cross_view.data['key_len']} keys"
    )

    # 4. Export standalone offline HTML
    out_dir = Path("examples/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "transformer_attention_explorer.html"
    view.save(out_file)
    print(f"\n4. Saved interactive offline HTML to: {out_file}")
    print(
        "   Open this file in any browser to inspect head routing, query tokens, and probabilities!"
    )


if __name__ == "__main__":
    main()
