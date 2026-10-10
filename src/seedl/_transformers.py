"""Transformer attention and sequence model visualization."""

from __future__ import annotations

from typing import Any

from .exceptions import SeeDLError


def inspect_attention(
    attention_weights: Any,
    *,
    tokens: list[str] | None = None,
    head: int | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    """Inspect transformer multi-head self-attention weights and query-key alignment."""
    # Convert weights to nested lists
    raw = attention_weights
    if hasattr(raw, "detach") and hasattr(raw, "cpu") and hasattr(raw, "tolist"):
        raw = raw.detach().cpu().tolist()
    elif hasattr(raw, "tolist"):
        raw = raw.tolist()

    # Determine rank
    dims = []
    curr = raw
    while isinstance(curr, list) and curr:
        dims.append(len(curr))
        curr = curr[0]

    rank = len(dims)
    if rank < 2:
        raise SeeDLError("Attention weights must be at least 2-dimensional (seq_len × seq_len).")

    # If 4D (batch, heads, seq, seq), take first batch item
    if rank == 4:
        raw = raw[0]
        rank = 3
        dims = dims[1:]

    # If 3D (heads, seq, seq), handle head selection or average
    num_heads = dims[0] if rank == 3 else 1
    seq_len = dims[-1]

    if rank == 3:
        if head is not None:
            if not (0 <= head < num_heads):
                raise SeeDLError(f"head index {head} out of range for {num_heads} attention heads.")
            selected_matrix = raw[head]
            active_head_str = f"Head {head}"
        else:
            # Average across heads
            selected_matrix = [
                [
                    sum(raw[h][i][j] for h in range(num_heads)) / float(num_heads)
                    for j in range(seq_len)
                ]
                for i in range(seq_len)
            ]
            active_head_str = f"Average of {num_heads} heads"
    else:
        selected_matrix = raw
        active_head_str = "Single head"

    token_names = [tokens[i] if tokens and i < len(tokens) else f"T{i}" for i in range(seq_len)]

    # Verify softmax property (rows sum to ~1.0)
    is_softmax = True
    for row in selected_matrix:
        row_sum = sum(float(v) for v in row)
        if abs(row_sum - 1.0) > 0.05:
            is_softmax = False
            break

    # Check for causal mask (upper triangle is zero)
    is_causal = True
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if abs(float(selected_matrix[i][j])) > 1e-5:
                is_causal = False
                break
        if not is_causal:
            break

    summary = (
        f"Transformer Attention: {seq_len} tokens · {active_head_str} ({num_heads} total heads)"
    )
    notes = [
        "Query-Key attention heatmap: Rows represent Queries (from), Columns represent Keys (to).",
    ]
    if is_softmax:
        notes.append("Values form a valid probability distribution (rows sum to ~1.0).")
    if is_causal:
        notes.append(
            "Causal masking detected: Tokens can only attend to preceding tokens (lower triangular)."
        )
    if tokens:
        notes.append(f"Sequence: {' '.join(token_names)}")

    data = {
        "matrix": selected_matrix,
        "tokens": token_names,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "selected_head": active_head_str,
        "active_head": active_head_str,
        "is_causal": is_causal,
        "shape": tuple(dims),
    }

    return data, summary, notes
