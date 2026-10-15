"""Transformer attention and sequence model visualization.

Provides genuine capture and visualization of multi-head self-attention,
cross-attention, query-key routing, head comparisons, and token attention matrices.
"""

from __future__ import annotations

import html
import json
import math
from typing import Any

from .exceptions import MissingDependencyError, SeeDLError


def _torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


def inspect_attention(
    attention_weights_or_model: Any,
    *,
    input: Any = None,
    tokens: list[str] | None = None,
    query_tokens: list[str] | None = None,
    key_tokens: list[str] | None = None,
    head: int | None = None,
    compare_heads: tuple[int, int] | None = None,
    batch_index: int = 0,
    is_batched: bool | None = None,
    mask: Any = None,
    max_seq_len: int = 256,
    max_heads: int = 32,
    theme: str = "scientific",
) -> tuple[dict[str, Any], str, list[str]]:
    """Inspect transformer multi-head self-attention weights and query-key alignment.

    Parameters:
        attention_weights_or_model: Attention weight tensor [heads, L, S] or [B, heads, L, S],
            or torch.nn.MultiheadAttention module (with input=...).
        input: Model input when passing a MultiheadAttention module.
        tokens: Token labels for square self-attention sequences.
        query_tokens: Token labels for query sequence (rows).
        key_tokens: Token labels for key sequence (columns).
        head: Selected attention head index (default 0 or averaged).
        compare_heads: Optional tuple of (head_a, head_b) to compare.
        batch_index: Batch index to inspect for 4D batched tensors (default 0).
        is_batched: Explicitly disambiguate 3D tensors: True for [B, L, S], False for [heads, L, S].
        mask: Optional attention mask matrix [L, S] or key mask [S].
        max_seq_len: Bounded sequence length limit (default 256).
        max_heads: Maximum heads to capture/preview (default 32).
        theme: UI theme ('scientific', 'light', 'dark').
    """
    torch = _torch()

    raw = attention_weights_or_model
    is_mha_module = torch is not None and isinstance(raw, torch.nn.MultiheadAttention)
    capture_note: str | None = None

    # Handle torch.nn.MultiheadAttention module
    if is_mha_module:
        if input is None:
            raise SeeDLError(
                "Inspecting torch.nn.MultiheadAttention requires input arguments: "
                "sd.attention(mha, input=(query, key, value))"
            )
        raw, capture_note = _run_mha_capture(raw, input)

    # Detach from PyTorch autograd if tensor
    if hasattr(raw, "detach"):
        raw = raw.detach()
    if hasattr(raw, "cpu"):
        raw = raw.cpu()
    if hasattr(raw, "tolist"):
        raw = raw.tolist()

    # Determine rank and dimensions
    dims = []
    curr = raw
    while isinstance(curr, list) and curr:
        dims.append(len(curr))
        curr = curr[0]

    rank = len(dims)
    if rank < 2:
        raise SeeDLError(
            "Attention weights must be at least 2-dimensional (query_tokens × key_tokens)."
        )

    # Batch selection for 4D [B, heads, L, S] or 3D [B, L, S]
    b_idx = 0
    b_size = 1
    is_averaged = False
    disambiguation_note: str | None = None

    if rank == 4:
        b_size = dims[0]
        b_idx = min(max(0, batch_index), b_size - 1)
        raw = raw[b_idx]
        rank = 3
        dims = dims[1:]
    elif rank == 3:
        # Disambiguate 3D tensor: [B, L, S] vs [heads, L, S]
        treat_as_batched = False
        if is_batched is True:
            treat_as_batched = True
        elif is_batched is False:
            treat_as_batched = False
        else:
            if is_mha_module or batch_index > 0 or (dims[0] > max_heads and dims[1] == dims[2]):
                treat_as_batched = True

        if treat_as_batched:
            b_size = dims[0]
            b_idx = min(max(0, batch_index), b_size - 1)
            raw = raw[b_idx]
            rank = 2
            dims = dims[1:]
            is_averaged = True
            disambiguation_note = (
                f"Disambiguated 3D tensor as batched attention [batch={b_size}, queries={dims[0]}, keys={dims[1]}]. "
                f"Displaying batch index {b_idx} (averaged across heads)."
            )
        else:
            disambiguation_note = (
                f"Disambiguated 3D tensor as multi-head attention [heads={dims[0]}, queries={dims[1]}, keys={dims[2]}]. "
                "Specify is_batched=True if dimension 0 represents batch."
            )

    # 3D: [heads, seq_q, seq_k]
    if rank == 3:
        num_heads = dims[0]
        query_len = dims[1]
        key_len = dims[2]
        all_heads_raw = raw[:max_heads]
        num_heads_captured = len(all_heads_raw)
    else:
        # 2D: [seq_q, seq_k]
        num_heads = 1
        num_heads_captured = 1
        query_len = dims[0]
        key_len = dims[1]
        all_heads_raw = [raw]
        is_averaged = is_mha_module or is_averaged

    # Bounded sequence length
    truncated_seq = False
    if query_len > max_seq_len or key_len > max_seq_len:
        truncated_seq = True
        q_lim = min(query_len, max_seq_len)
        k_lim = min(key_len, max_seq_len)
        all_heads_raw = [
            [[head_mat[i][j] for j in range(k_lim)] for i in range(q_lim)]
            for head_mat in all_heads_raw
        ]
        query_len = q_lim
        key_len = k_lim

    # Compute averaged attention matrix across heads
    avg_matrix: list[list[float]] = []
    for i in range(query_len):
        avg_row: list[float] = []
        for j in range(key_len):
            avg_val = sum(all_heads_raw[h][i][j] for h in range(num_heads_captured)) / float(
                num_heads_captured
            )
            avg_row.append(avg_val)
        avg_matrix.append(avg_row)

    # Resolve active head
    if head is not None:
        if not (0 <= head < num_heads_captured):
            raise SeeDLError(
                f"Head index {head} is out of range for {num_heads_captured} captured attention heads."
            )
        selected_matrix = all_heads_raw[head]
        active_head_str = f"Head {head}"
        selected_head_idx = head
    else:
        if num_heads_captured > 1:
            selected_matrix = avg_matrix
            active_head_str = f"Average of {num_heads_captured} heads"
            selected_head_idx = -1  # -1 represents Average
        else:
            selected_matrix = all_heads_raw[0]
            active_head_str = "Single head" if not is_averaged else "Averaged head"
            selected_head_idx = 0

    # Resolve token names
    if query_tokens is not None:
        q_tokens = [query_tokens[i] if i < len(query_tokens) else f"Q{i}" for i in range(query_len)]
    elif tokens is not None:
        q_tokens = [tokens[i] if i < len(tokens) else f"T{i}" for i in range(query_len)]
    else:
        q_tokens = [str(i) for i in range(query_len)]

    if key_tokens is not None:
        k_tokens = [key_tokens[j] if j < len(key_tokens) else f"K{j}" for j in range(key_len)]
    elif tokens is not None:
        k_tokens = [tokens[j] if j < len(tokens) else f"T{j}" for j in range(key_len)]
    else:
        k_tokens = [str(j) for j in range(key_len)]

    # Compute statistics and scientific checks
    is_square = query_len == key_len
    is_cross_attention = not is_square

    # Parse optional attention mask
    mask_grid: list[list[bool]] | None = None
    if mask is not None:
        m_raw = mask
        if hasattr(m_raw, "detach"):
            m_raw = m_raw.detach().cpu()
        if hasattr(m_raw, "tolist"):
            m_raw = m_raw.tolist()

        if isinstance(m_raw, (list, tuple)) and m_raw:
            if not isinstance(m_raw[0], (list, tuple)):
                # 1D key padding mask [key_len]
                m_1d = [
                    bool(v is True or (isinstance(v, (int, float)) and v <= -1e4))
                    for v in m_raw[:key_len]
                ]
                while len(m_1d) < key_len:
                    m_1d.append(False)
                mask_grid = [list(m_1d) for _ in range(query_len)]
            else:
                # 2D mask [query_len, key_len]
                mask_grid = []
                for i in range(query_len):
                    if i < len(m_raw) and isinstance(m_raw[i], (list, tuple)):
                        row = [
                            bool(v is True or (isinstance(v, (int, float)) and v <= -1e4))
                            for v in m_raw[i][:key_len]
                        ]
                        while len(row) < key_len:
                            row.append(False)
                        mask_grid.append(row)
                    else:
                        mask_grid.append([False] * key_len)

    # Verify probability distribution (row sums) and non-negativity
    is_softmax = True
    has_negatives = False

    for row in selected_matrix:
        row_sum = 0.0
        for val in row:
            f_val = float(val)
            if f_val < 0.0:
                has_negatives = True
            row_sum += f_val
        if abs(row_sum - 1.0) > 0.05:
            is_softmax = False

    mean_entropy: float | None = None
    if is_softmax and not has_negatives:
        row_entropies: list[float] = []
        for row in selected_matrix:
            r_sum = sum(float(val) for val in row)
            entropy = 0.0
            if r_sum > 0.0:
                for val in row:
                    p = max(0.0, float(val)) / r_sum
                    if p > 1e-12:
                        entropy -= p * math.log2(p)
            row_entropies.append(entropy)
        mean_entropy = sum(row_entropies) / len(row_entropies) if row_entropies else 0.0

    # Check for causal mask (only applicable to square self-attention)
    is_causal = is_square
    if is_causal:
        for i in range(query_len):
            for j in range(i + 1, key_len):
                if abs(float(selected_matrix[i][j])) > 1e-4:
                    is_causal = False
                    break
            if not is_causal:
                break

    # Individual head statistics
    head_stats = []
    for h_idx, h_mat in enumerate(all_heads_raw):
        vals = [float(v) for r in h_mat for v in r if math.isfinite(v)]
        min_v = min(vals) if vals else 0.0
        max_v = max(vals) if vals else 0.0
        mean_v = sum(vals) / len(vals) if vals else 0.0
        head_stats.append(
            {
                "head": h_idx,
                "min": min_v,
                "max": max_v,
                "mean": mean_v,
            }
        )

    # Multi-head comparison payload if requested
    head_comparison_payload = None
    if compare_heads is not None:
        ha, hb = compare_heads
        head_comparison_payload = compare_attention_heads(
            all_heads_raw, ha, hb, q_tokens, k_tokens, is_averaged
        )

    summary = (
        f"Transformer Attention: {query_len} queries × {key_len} keys · "
        f"{active_head_str} ({num_heads} total heads)"
    )

    notes = [
        "Query-Key attention heatmap: Rows represent Queries (from), Columns represent Keys (to).",
        "Attention weights represent observational query-key routing probabilities; they do not prove causal feature importance or explain model decisions.",
    ]
    if capture_note:
        notes.append(capture_note)
    if disambiguation_note:
        notes.append(disambiguation_note)
    if mask_grid is not None:
        num_masked = sum(sum(1 for cell in r if cell) for r in mask_grid)
        if num_masked > 0:
            notes.append(
                f"Supplied attention mask applied: {num_masked} of {query_len * key_len} query-key pairs masked."
            )
    if is_cross_attention:
        notes.append(
            f"Cross-attention detected: query sequence length ({query_len}) differs from key sequence length ({key_len})."
        )
    if is_softmax and not has_negatives:
        notes.append(
            "Values form a valid probability distribution (rows sum to ~1.0 with non-negative weights)."
        )
    elif has_negatives or not is_softmax:
        notes.append(
            "Attention weights contain negative values or do not sum to 1.0; displaying genuine unnormalized weights."
        )
    if is_causal:
        notes.append(
            "Causal masking detected: Tokens can only attend to preceding tokens (lower triangular)."
        )
    if is_averaged:
        notes.append(
            "Attention weights are averaged across heads; individual heads were not exposed."
        )
    if truncated_seq:
        notes.append(
            f"Bounded preview: sequence truncated to {max_seq_len} tokens to ensure responsive rendering."
        )

    data = {
        "presentation": "attention",
        "theme": theme,
        "matrix": selected_matrix,
        "all_heads": all_heads_raw,
        "avg_matrix": avg_matrix,
        "tokens": q_tokens,
        "query_tokens": q_tokens,
        "key_tokens": k_tokens,
        "seq_len": key_len,
        "query_len": query_len,
        "key_len": key_len,
        "num_heads": num_heads,
        "captured_heads": num_heads_captured,
        "selected_head_idx": selected_head_idx,
        "selected_head": active_head_str,
        "active_head": active_head_str,
        "is_causal": is_causal,
        "is_softmax": is_softmax,
        "is_averaged": is_averaged,
        "is_cross_attention": is_cross_attention,
        "has_negatives": has_negatives,
        "mean_entropy": mean_entropy,
        "head_stats": head_stats,
        "head_comparison": head_comparison_payload,
        "mask": mask_grid,
        "shape": tuple(dims),
    }

    return data, summary, notes


def _run_mha_capture(mha_module: Any, example_input: Any) -> tuple[Any, str]:
    """Execute PyTorch nn.MultiheadAttention safely and capture genuine returned attention weights."""
    torch = _torch()
    assert torch is not None

    # Handle input arguments (query, key, value)
    if isinstance(example_input, (tuple, list)):
        if len(example_input) == 3:
            q, k, v = example_input
        elif len(example_input) == 1:
            q = k = v = example_input[0]
        else:
            raise SeeDLError(
                "MultiheadAttention input must be (query, key, value) or a single tensor for self-attention."
            )
    else:
        q = k = v = example_input

    orig_training_modes = [(m, m.training) for m in mha_module.modules()]
    mha_module.eval()

    try:
        with torch.no_grad():
            try:
                output = mha_module(q, k, v, average_attn_weights=False)
            except TypeError:
                output = mha_module(q, k, v)
    except Exception as exc:
        raise SeeDLError(f"Failed to execute MultiheadAttention forward pass: {exc}") from exc
    finally:
        for m, orig_mode in orig_training_modes:
            m.training = orig_mode

    # In PyTorch, MultiheadAttention returns (attn_output, attn_weights)
    if isinstance(output, tuple) and len(output) >= 2:
        weights = output[1]
        if weights is None:
            raise SeeDLError(
                "MultiheadAttention forward pass returned None for attention weights. "
                "Ensure need_weights=True is enabled when calling the module."
            )
        w_shape = tuple(weights.shape)
        if weights.ndim == 3:
            note = f"Observed attention weights with shape {w_shape} (PyTorch default average_attn_weights=True)."
        else:
            note = f"Observed per-head attention weights with shape {w_shape}."
        return weights, note

    raise SeeDLError(
        "MultiheadAttention did not return an attention weight tensor. "
        "Standard PyTorch nn.MultiheadAttention returns (output, weights)."
    )


def compare_attention_heads(
    all_heads: list[list[list[float]]],
    head_a: int,
    head_b: int,
    query_tokens: list[str],
    key_tokens: list[str],
    is_averaged: bool,
) -> dict[str, Any]:
    """Compare two attention heads from the same multi-head tensor or compatible tensors."""
    if is_averaged or len(all_heads) <= 1:
        return {
            "available": False,
            "reason": "Individual head comparison is unavailable on averaged or single-head attention.",
        }

    num_heads = len(all_heads)
    if not (0 <= head_a < num_heads) or not (0 <= head_b < num_heads):
        raise SeeDLError(
            f"Invalid head comparison indices ({head_a}, {head_b}); tensor has {num_heads} heads."
        )

    mat_a = all_heads[head_a]
    mat_b = all_heads[head_b]

    qh, kw = len(mat_a), len(mat_a[0]) if mat_a else 0

    diff_matrix: list[list[float]] = []
    abs_diffs = []
    sq_diffs = []

    dot_prod = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for i in range(qh):
        diff_row = []
        for j in range(kw):
            va = mat_a[i][j]
            vb = mat_b[i][j]
            d = vb - va
            diff_row.append(d)
            abs_diffs.append(abs(d))
            sq_diffs.append(d * d)
            dot_prod += va * vb
            norm_a_sq += va * va
            norm_b_sq += vb * vb
        diff_matrix.append(diff_row)

    mae = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
    frobenius_norm = math.sqrt(sum(sq_diffs))
    max_diff = max(abs_diffs) if abs_diffs else 0.0
    cosine_sim = (
        dot_prod / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))
        if norm_a_sq > 0 and norm_b_sq > 0
        else 0.0
    )

    return {
        "available": True,
        "head_a": head_a,
        "head_b": head_b,
        "matrix_a": mat_a,
        "matrix_b": mat_b,
        "diff_matrix": diff_matrix,
        "mae": mae,
        "frobenius_norm": frobenius_norm,
        "max_abs_diff": max_diff,
        "cosine_similarity": cosine_sim,
        "query_tokens": query_tokens,
        "key_tokens": key_tokens,
    }


# ==============================================================================
# RENDERERS: PLOTLY, SVG, MATPLOTLIB, HTML
# ==============================================================================


def plotly_attention(view: Any) -> Any:
    """Render interactive Plotly Transformer Attention explorer with head selection and query token inspection."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise MissingDependencyError("Plotly attention rendering requires Plotly.") from exc

    data = view.data
    theme = data.get("theme", "scientific")
    bg = "#0f172a" if theme in {"dark", "scientific"} else "#f8fafc"
    fg = "#f8fafc" if theme in {"dark", "scientific"} else "#0f172a"
    grid_col = "#334155" if theme in {"dark", "scientific"} else "#e2e8f0"

    q_toks = data.get("query_tokens", data.get("tokens", []))
    k_toks = data.get("key_tokens", data.get("tokens", []))
    all_heads = data.get("all_heads", [data.get("matrix", [])])
    avg_matrix = data.get("avg_matrix", data.get("matrix", []))
    selected_matrix = data.get("matrix", [])

    is_signed = data.get("has_negatives", False)
    cs = "RdBu_r" if is_signed else "Blues"

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=[
            f"Attention Matrix · {data.get('selected_head', 'Head 0')}",
            "Query Token Attention Profile (Query 0)",
        ],
        horizontal_spacing=0.12,
    )

    # 1. Main Heatmap
    fig.add_trace(
        go.Heatmap(
            z=selected_matrix,
            x=k_toks,
            y=q_toks,
            colorscale=cs,
            zmin=-1.0 if is_signed else 0.0,
            zmax=1.0,
            colorbar=dict(title="Probability", x=0.62, len=0.85),
            hovertemplate=(
                "<b>Query:</b> %{y}<br>"
                "<b>Key:</b> %{x}<br>"
                "<b>Attention Weight:</b> %{z:.4f}<extra></extra>"
            ),
            name="Attention",
        ),
        row=1,
        col=1,
    )

    # 2. Query 0 bar chart
    q0_weights = selected_matrix[0] if selected_matrix else []
    fig.add_trace(
        go.Bar(
            x=k_toks,
            y=q0_weights,
            marker=dict(color="#38bdf8"),
            hovertemplate="Key: %{x}<br>Attention: %{y:.4f}<extra></extra>",
            name="Query 0",
        ),
        row=1,
        col=2,
    )

    # Dropdown for Head selection
    updatemenus = []
    if len(all_heads) > 1:
        head_buttons = []
        for h_idx, h_mat in enumerate(all_heads[:24]):
            head_buttons.append(
                dict(
                    method="update",
                    label=f"Head {h_idx}",
                    args=[
                        {
                            "z": [h_mat, None],
                            "y": [q_toks, [h_mat[0] if h_mat else []]],
                        },
                        {
                            "subplot_titles": [
                                f"Attention Matrix · Head {h_idx}",
                                "Query Token Attention Profile (Query 0)",
                            ]
                        },
                    ],
                )
            )
        # Average head button
        head_buttons.append(
            dict(
                method="update",
                label=f"Average ({len(all_heads)} heads)",
                args=[
                    {
                        "z": [avg_matrix, None],
                        "y": [q_toks, [avg_matrix[0] if avg_matrix else []]],
                    },
                    {
                        "subplot_titles": [
                            f"Attention Matrix · Average of {len(all_heads)} heads",
                            "Query Token Attention Profile (Query 0)",
                        ]
                    },
                ],
            )
        )
        updatemenus.append(
            dict(
                buttons=head_buttons,
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.18,
                yanchor="top",
                bgcolor=bg,
                font=dict(color=fg, size=11),
                bordercolor=grid_col,
            )
        )

    entropy_val = data.get("mean_entropy")
    entropy_str = (
        f" · Entropy: {entropy_val:.2f} bits"
        if entropy_val is not None
        else " · Entropy: N/A (unnormalized/logits)"
    )
    fig.update_layout(
        title=dict(
            text=f"<b>{view.title}</b><br><sup>{view.summary}{entropy_str}</sup>",
            x=0.04,
            y=0.96,
            font=dict(size=18, color=fg),
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Inter, system-ui, sans-serif", color=fg),
        height=540,
        margin=dict(l=60, r=40, t=110, b=60),
        updatemenus=updatemenus,
    )

    fig.update_yaxes(autorange="reversed", title_text="Query Tokens (From)", row=1, col=1)
    fig.update_xaxes(title_text="Key Tokens (To)", row=1, col=1)
    fig.update_xaxes(title_text="Key Tokens", row=1, col=2)
    fig.update_yaxes(title_text="Attention Weight", row=1, col=2, range=[0, 1.05])

    return fig


def plotly_attention_comparison(view: Any) -> Any:
    """Render Plotly head comparison (Head A, Head B, and Difference matrix)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise MissingDependencyError("Attention comparison requires Plotly.") from exc

    data = view.data
    cmp = data.get("head_comparison")
    if not cmp or not cmp.get("available"):
        raise SeeDLError("Head comparison data is not available for this view.")

    theme = data.get("theme", "scientific")
    bg = "#0f172a" if theme in {"dark", "scientific"} else "#f8fafc"
    fg = "#f8fafc" if theme in {"dark", "scientific"} else "#0f172a"

    ha = cmp["head_a"]
    hb = cmp["head_b"]
    mat_a = cmp["matrix_a"]
    mat_b = cmp["matrix_b"]
    diff = cmp["diff_matrix"]
    q_toks = cmp["query_tokens"]
    k_toks = cmp["key_tokens"]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"Head {ha}",
            f"Head {hb}",
            f"Difference (Head {hb} - Head {ha})",
        ],
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Heatmap(
            z=mat_a, x=k_toks, y=q_toks, colorscale="Blues", zmin=0, zmax=1, showscale=False
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=mat_b, x=k_toks, y=q_toks, colorscale="Blues", zmin=0, zmax=1, showscale=False
        ),
        row=1,
        col=2,
    )

    max_d = max(cmp.get("max_abs_diff", 0.5), 0.01)
    fig.add_trace(
        go.Heatmap(
            z=diff,
            x=k_toks,
            y=q_toks,
            colorscale="RdBu_r",
            zmin=-max_d,
            zmax=max_d,
            colorbar=dict(title="Diff", x=0.98, len=0.8),
        ),
        row=1,
        col=3,
    )

    for c in range(1, 4):
        fig.update_yaxes(
            autorange="reversed", title_text="Query Tokens" if c == 1 else "", row=1, col=c
        )
        fig.update_xaxes(title_text="Key Tokens", row=1, col=c)

    fig.update_layout(
        title=dict(
            text=f"<b>Attention Head Comparison: Head {ha} vs Head {hb}</b><br><sup>MAE: {cmp['mae']:.4f} · Frobenius Norm: {cmp['frobenius_norm']:.4f} · Cosine Sim: {cmp['cosine_similarity']:.4f}</sup>",
            x=0.04,
            y=0.96,
            font=dict(size=18, color=fg),
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(family="Inter, system-ui, sans-serif", color=fg),
        height=480,
        margin=dict(l=50, r=40, t=100, b=50),
    )

    return fig


def svg_attention(view: Any) -> str:
    """Render clean static SVG of the transformer attention matrix."""
    data = view.data
    theme = data.get("theme", "scientific")
    bg = "#0f172a" if theme in {"dark", "scientific"} else "#f8fafc"
    fg = "#f8fafc" if theme in {"dark", "scientific"} else "#0f172a"
    muted = "#94a3b8" if theme in {"dark", "scientific"} else "#64748b"

    matrix = data.get("matrix", [])
    q_toks = data.get("query_tokens", data.get("tokens", []))
    k_toks = data.get("key_tokens", data.get("tokens", []))

    qh = len(matrix)
    kw = len(matrix[0]) if qh > 0 else 0

    cell = min(36.0, 480.0 / max(kw, qh, 1))
    gx0 = 120
    gy0 = 120

    rects = []
    labels_q = []
    labels_k = []

    for i in range(qh):
        # Query token label on left
        q_label = html.escape(q_toks[i] if i < len(q_toks) else str(i))
        labels_q.append(
            f'<text x="{gx0 - 10}" y="{gy0 + i * cell + cell * 0.7:.1f}" text-anchor="end" font-family="monospace" font-size="11" fill="{fg}">{q_label}</text>'
        )

        for j in range(kw):
            val = float(matrix[i][j])
            norm = max(0.0, min(1.0, val))
            # Blue colormap
            r_col = int(240 - norm * 200)
            g_col = int(245 - norm * 150)
            b_col = int(255 - norm * 50)
            fill = (
                f"rgb({r_col},{g_col},{b_col})"
                if theme != "dark"
                else f"rgb({int(15 + norm * 40)},{int(30 + norm * 140)},{int(80 + norm * 175)})"
            )

            k_title = html.escape(k_toks[j] if j < len(k_toks) else str(j))
            rects.append(
                f'<rect x="{gx0 + j * cell:.1f}" y="{gy0 + i * cell:.1f}" width="{cell - 1:.1f}" height="{cell - 1:.1f}" fill="{fill}">'
                f"<title>({q_label} → {k_title}): {val:.4f}</title></rect>"
            )

    for j in range(kw):
        k_label = html.escape(k_toks[j] if j < len(k_toks) else str(j))
        labels_k.append(
            f'<text x="{gx0 + j * cell + cell * 0.5:.1f}" y="{gy0 - 10}" text-anchor="middle" font-family="monospace" font-size="11" fill="{fg}">{k_label}</text>'
        )

    width = max(680, int(gx0 + kw * cell + 60))
    height = max(400, int(gy0 + qh * cell + 60))

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        f'<text x="40" y="38" font-family="system-ui, sans-serif" font-size="12" fill="{muted}" font-weight="600">SeeDL / TRANSFORMER ATTENTION</text>',
        f'<text x="40" y="70" font-family="system-ui, sans-serif" font-size="20" fill="{fg}" font-weight="700">{html.escape(view.title)}</text>',
        f'<text x="40" y="92" font-family="system-ui, sans-serif" font-size="13" fill="{muted}">{html.escape(view.summary)}</text>',
        "".join(labels_k),
        "".join(labels_q),
        "".join(rects),
        "</svg>",
    ]

    return "\n".join(svg_lines)


def svg_attention_comparison(view: Any) -> str:
    """Render static SVG of head comparison."""
    data = view.data
    cmp = data.get("head_comparison", {})
    width = 720
    height = 360
    bg = "#0f172a"
    fg = "#f8fafc"
    muted = "#94a3b8"

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">
      <rect width="{width}" height="{height}" fill="{bg}"/>
      <text x="40" y="40" fill="{muted}" font-size="12" font-weight="600">SeeDL / HEAD COMPARISON</text>
      <text x="40" y="75" fill="{fg}" font-size="20" font-weight="700">Head {cmp.get("head_a", 0)} vs Head {cmp.get("head_b", 1)}</text>
      <text x="40" y="105" fill="{muted}" font-size="13">MAE: {cmp.get("mae", 0.0):.4f} · Frobenius Norm: {cmp.get("frobenius_norm", 0.0):.4f} · Cosine Similarity: {cmp.get("cosine_similarity", 0.0):.4f}</text>
    </svg>"""


def matplotlib_attention(view: Any) -> Any:
    """Render Matplotlib figure of Transformer attention matrix and query distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise MissingDependencyError("Matplotlib rendering requires matplotlib.") from exc

    data = view.data
    matrix = data.get("matrix", [])
    q_toks = data.get("query_tokens", data.get("tokens", []))
    k_toks = data.get("key_tokens", data.get("tokens", []))

    fig, (ax_mat, ax_bar) = plt.subplots(
        1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )

    im = ax_mat.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax_mat, label="Attention Weight")
    ax_mat.set_title(data.get("selected_head", "Attention Matrix"), fontsize=11, fontweight="bold")

    if k_toks:
        ax_mat.set_xticks(range(len(k_toks)))
        ax_mat.set_xticklabels(k_toks, rotation=45, ha="right")
    if q_toks:
        ax_mat.set_yticks(range(len(q_toks)))
        ax_mat.set_yticklabels(q_toks)

    ax_mat.set_xlabel("Key Tokens (To)")
    ax_mat.set_ylabel("Query Tokens (From)")

    # Query 0 profile bar chart
    q0 = matrix[0] if matrix else []
    ax_bar.bar(range(len(q0)), q0, color="#0284c7")
    ax_bar.set_title("Query 0 Attention Profile", fontsize=11, fontweight="bold")
    ax_bar.set_xlabel("Key Token Index")
    ax_bar.set_ylabel("Attention Weight")
    ax_bar.set_ylim(0, 1.05)

    fig.suptitle(f"{view.title}\n{view.summary}", fontsize=12)
    fig.tight_layout()
    return fig


def attention_html(view: Any) -> str:
    """Generate self-contained, offline interactive HTML for Transformer Attention exploration."""
    data = view.data
    title = html.escape(view.title)
    summary = html.escape(view.summary)
    payload_json = (
        json.dumps(data).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title} — SeeDL Attention Explorer</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }}
    header {{
      margin-bottom: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }}
    .badge {{
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }}
    h1 {{ font-size: 22px; color: var(--text); margin-bottom: 6px; }}
    .summary {{ font-size: 14px; color: var(--muted); }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      align-items: center;
      margin-bottom: 24px;
      background: rgba(15, 23, 42, 0.6);
      padding: 12px 16px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .control-group {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    label {{ font-size: 12px; font-weight: 600; color: var(--muted); }}
    select, button {{
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 13px;
      cursor: pointer;
    }}
    select:focus, button:focus {{ outline: 2px solid var(--accent); }}
    .btn-active {{
      background: var(--accent);
      color: #0f172a;
      font-weight: 600;
      border-color: var(--accent);
    }}
    .view-grid {{
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 24px;
    }}
    @media (max-width: 900px) {{
      .view-grid {{ grid-template-columns: 1fr; }}
    }}
    .matrix-container {{
      background: #020617;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      overflow-x: auto;
    }}
    .token-table {{
      border-collapse: separate;
      border-spacing: 2px;
      margin: 0 auto;
    }}
    .cell {{
      width: 32px;
      height: 32px;
      text-align: center;
      border-radius: 2px;
      cursor: pointer;
      font-family: monospace;
      font-size: 10px;
      transition: transform 0.1s;
    }}
    .cell:hover {{
      transform: scale(1.15);
      outline: 2px solid var(--accent);
      z-index: 10;
    }}
    .row-hdr, .col-hdr {{
      font-family: monospace;
      font-size: 11px;
      color: var(--muted);
      padding: 4px;
    }}
    .profile-card {{
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .profile-card h3 {{
      font-size: 14px;
      color: var(--accent);
      border-bottom: 1px solid var(--border);
      padding-bottom: 6px;
    }}
    .bars-container {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 320px;
      overflow-y: auto;
    }}
    .bar-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }}
    .bar-lbl {{ width: 80px; font-family: monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text); }}
    .bar-track {{ flex: 1; height: 14px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--accent); border-radius: 3px; }}
    .bar-val {{ width: 45px; text-align: right; font-family: monospace; color: var(--muted); }}
    .hud {{
      margin-top: 12px;
      font-size: 13px;
      font-family: monospace;
      color: var(--accent);
      min-height: 20px;
    }}
    footer {{
      margin-top: 24px;
      font-size: 12px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      border-top: 1px solid var(--border);
      padding-top: 12px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">SeeDL · Attention Intelligence 0.4.0</div>
      <h1>{title}</h1>
      <div class="summary">{summary}</div>
    </header>

    <div class="controls">
      <div class="control-group">
        <label for="headSelect">Attention Head</label>
        <select id="headSelect"></select>
      </div>

      <div class="control-group">
        <label for="querySelect">Inspect Query Token</label>
        <select id="querySelect"></select>
      </div>
    </div>

    <div class="view-grid">
      <div class="matrix-container">
        <div id="tableWrapper"></div>
        <div id="hoverHud" class="hud">Hover over attention cells to inspect token alignment and weight</div>
      </div>

      <div class="profile-card">
        <h3 id="profileTitle">Query Attention Profile</h3>
        <div id="barsList" class="bars-container"></div>
        <div style="border-top:1px solid var(--border); padding-top:10px; font-size:12px; color:var(--muted);">
          <div>Entropy: <span id="stEntropy" style="color:var(--text); font-weight:600;">—</span> bits</div>
          <div>Causal Mask: <span id="stCausal" style="color:var(--text); font-weight:600;">—</span></div>
          <div>Softmax Valid: <span id="stSoftmax" style="color:var(--text); font-weight:600;">—</span></div>
        </div>
      </div>
    </div>

    <footer>
      <span>Exported by SeeDL</span>
      <span>Authentic Attention Weights · Observational Routing Probabilities</span>
    </footer>
  </div>

  <script>
    const DATA = {payload_json};
    let currentHeadIdx = DATA.selected_head_idx >= 0 ? DATA.selected_head_idx : 0;
    let currentQueryIdx = 0;

    function escapeHtml(str) {{
      if (str == null) return "";
      return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }}

    const headSelect = document.getElementById("headSelect");
    const querySelect = document.getElementById("querySelect");
    const tableWrapper = document.getElementById("tableWrapper");
    const barsList = document.getElementById("barsList");
    const hoverHud = document.getElementById("hoverHud");

    function init() {{
      const numH = DATA.captured_heads || 1;
      for (let h = 0; h < numH; h++) {{
        const opt = document.createElement("option");
        opt.value = h;
        opt.textContent = `Head ${{h}}`;
        headSelect.appendChild(opt);
      }}
      if (numH > 1) {{
        const optAvg = document.createElement("option");
        optAvg.value = -1;
        optAvg.textContent = `Average (${{numH}} heads)`;
        headSelect.appendChild(optAvg);
      }}
      headSelect.value = currentHeadIdx;

      DATA.query_tokens.forEach((tok, idx) => {{
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = `[${{idx}}] ${{tok}}`;
        querySelect.appendChild(opt);
      }});

      headSelect.addEventListener("change", (e) => {{
        currentHeadIdx = parseInt(e.target.value);
        render();
      }});

      querySelect.addEventListener("change", (e) => {{
        currentQueryIdx = parseInt(e.target.value);
        renderProfile();
      }});

      render();
    }}

    function getActiveMatrix() {{
      if (currentHeadIdx === -1) return DATA.avg_matrix;
      return DATA.all_heads[currentHeadIdx] || DATA.matrix;
    }}

    function render() {{
      const mat = getActiveMatrix();
      const qToks = DATA.query_tokens;
      const kToks = DATA.key_tokens;
      const qh = mat.length;
      const kw = qh > 0 ? mat[0].length : 0;

      // Render HTML matrix table
      let htmlStr = '<table class="token-table"><thead><tr><th></th>';
      for (let j = 0; j < kw; j++) {{
        htmlStr += `<th class="col-hdr">${{escapeHtml(kToks[j])}}</th>`;
      }}
      htmlStr += '</tr></thead><tbody>';

      for (let i = 0; i < qh; i++) {{
        htmlStr += `<tr><td class="row-hdr">${{escapeHtml(qToks[i])}}</td>`;
        for (let j = 0; j < kw; j++) {{
          const val = mat[i][j];
          const isMasked = DATA.mask && DATA.mask[i] && DATA.mask[i][j];
          const norm = Math.max(0, Math.min(1, val));
          const r = isMasked ? 40 : Math.round(15 + norm * 40);
          const g = isMasked ? 40 : Math.round(30 + norm * 140);
          const b = isMasked ? 50 : Math.round(80 + norm * 175);
          htmlStr += `<td class="cell" style="background:rgb(${{r}},${{g}},${{b}});"
                       onmouseover="showHover(${{i}}, ${{j}}, ${{val}})"
                       onclick="selectQuery(${{i}})"></td>`;
        }}
        htmlStr += '</tr>';
      }}
      htmlStr += '</tbody></table>';
      tableWrapper.innerHTML = htmlStr;

      document.getElementById("stEntropy").textContent = DATA.mean_entropy != null ? (DATA.mean_entropy.toFixed(2) + " bits") : "N/A (unnormalized/logits)";
      document.getElementById("stCausal").textContent = DATA.is_causal ? "Yes (Lower triangular)" : "No";
      document.getElementById("stSoftmax").textContent = DATA.is_softmax ? "Yes (~1.0 per row)" : "Unnormalized/Logits";

      renderProfile();
    }}

    window.showHover = function(qIdx, kIdx, val) {{
      const qTok = DATA.query_tokens[qIdx];
      const kTok = DATA.key_tokens[kIdx];
      const isMasked = DATA.mask && DATA.mask[qIdx] && DATA.mask[qIdx][kIdx];
      const maskStr = isMasked ? " · [Masked]" : "";
      hoverHud.textContent = `Query [${{qIdx}}] "${{qTok}}" → Key [${{kIdx}}] "${{kTok}}" · Weight: ${{val.toFixed(4)}} (${{(val * 100).toFixed(1)}}%)${{maskStr}}`;
    }};

    window.selectQuery = function(qIdx) {{
      currentQueryIdx = qIdx;
      querySelect.value = qIdx;
      renderProfile();
    }};

    function renderProfile() {{
      const mat = getActiveMatrix();
      const qTok = DATA.query_tokens[currentQueryIdx];
      const kToks = DATA.key_tokens;
      const row = mat[currentQueryIdx] || [];

      document.getElementById("profileTitle").textContent = `Query [${{currentQueryIdx}}] "${{qTok}}" Attention Distribution`;
      barsList.innerHTML = "";

      row.forEach((val, kIdx) => {{
        const rowDiv = document.createElement("div");
        rowDiv.className = "bar-row";

        const lbl = document.createElement("div");
        lbl.className = "bar-lbl";
        lbl.textContent = `[${{kIdx}}] ${{kToks[kIdx]}}`;

        const track = document.createElement("div");
        track.className = "bar-track";
        const fill = document.createElement("div");
        fill.className = "bar-fill";
        fill.style.width = Math.max(0, Math.min(100, val * 100)) + "%";
        track.appendChild(fill);

        const valLbl = document.createElement("div");
        valLbl.className = "bar-val";
        valLbl.textContent = val.toFixed(3);

        rowDiv.appendChild(lbl);
        rowDiv.appendChild(track);
        rowDiv.appendChild(valLbl);
        barsList.appendChild(rowDiv);
      }});
    }}

    init();
  </script>
</body>
</html>"""
    return doc


def inspect_vit_attention(
    attention_weights: Any,
    patch_grid: tuple[int, int],
    *,
    from_token: int = 0,
    has_cls: bool = True,
    head: int | None = None,
    theme: str = "scientific",
) -> tuple[dict[str, Any], str, list[str]]:
    """Reshape Vision Transformer token attention slice into a 2D spatial patch attention map."""
    data, summary, notes = inspect_attention(attention_weights, head=head, theme=theme)
    mat = data["matrix"]
    gh, gw = patch_grid
    expected_patches = gh * gw

    if from_token >= len(mat):
        raise SeeDLError(f"from_token {from_token} out of range for sequence length {len(mat)}.")

    attn_row = mat[from_token]
    # If has_cls, patch tokens start at index 1
    start_idx = 1 if has_cls else 0
    patch_vals = attn_row[start_idx : start_idx + expected_patches]

    if len(patch_vals) < expected_patches:
        raise SeeDLError(
            f"Attention vector length ({len(patch_vals)}) does not match patch grid {gh}×{gw} ({expected_patches} patches)."
        )

    # Reshape into [gh, gw]
    grid_2d = [patch_vals[r * gw : (r + 1) * gw] for r in range(gh)]

    from ._features import _compute_channel_stats

    stats = _compute_channel_stats(grid_2d)

    vit_payload = {
        "presentation": "feature_maps",
        "theme": theme,
        "layers": [
            {
                "name": f"ViT Patch Attention (from Token {from_token})",
                "module_name": "VisionTransformer",
                "type": "PatchAttention",
                "original_shape": (1, 1, gh, gw),
                "captured_shape": (1, 1, gh, gw),
                "total_channels": 1,
                "captured_channels": 1,
                "batch_size": 1,
                "batch_index": 0,
                "downsampled": False,
                "original_spatial": (gh, gw),
                "channels": [
                    {
                        "index": 0,
                        "grid": grid_2d,
                        "h": gh,
                        "w": gw,
                        "stats": stats,
                    }
                ],
                "layer_stats": {
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": stats["mean"],
                },
            }
        ],
        "selected_layer_idx": 0,
        "selected_channel_idx": 0,
        "total_layers": 1,
        "batch_index": 0,
        "normalization_modes": ["per_channel", "layer_global", "zero_centered"],
    }

    vit_summary = (
        f"ViT Patch Attention: {gh}×{gw} grid from Token {from_token} · {data['active_head']}"
    )
    vit_notes = [
        f"Vision Transformer attention vector from token {from_token} reshaped to {gh}×{gw} spatial patch geometry.",
        "Displays where the selected token (e.g. [CLS]) routes its attention across image patches.",
    ]
    return vit_payload, vit_summary, vit_notes
