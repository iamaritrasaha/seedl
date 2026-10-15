"""Scientific correctness and integration tests for Transformer Attention Explorer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import seedl as sd
from seedl._transformers import compare_attention_heads, inspect_attention
from seedl.exceptions import SeeDLError


def test_attention_known_softmax_matrix_and_orientation():
    # 3x3 attention matrix where queries attend to keys
    # Row 0: attends strongly to Key 0 (0.8) and Key 1 (0.2)
    # Row 1: attends equally to all keys (1/3 each)
    # Row 2: attends strongly to Key 2 (1.0)
    matrix = [
        [0.8, 0.2, 0.0],
        [1.0 / 3, 1.0 / 3, 1.0 / 3],
        [0.0, 0.0, 1.0],
    ]
    tokens = ["TokA", "TokB", "TokC"]

    data, summary, notes = inspect_attention(matrix, tokens=tokens)

    assert data["query_len"] == 3
    assert data["key_len"] == 3
    assert data["is_softmax"] is True
    assert data["is_causal"] is False

    # Check query 0 values
    assert data["matrix"][0][0] == 0.8
    assert data["matrix"][0][1] == 0.2
    assert data["matrix"][0][2] == 0.0

    # Row 2 has zero entropy (pure peak on key 2)
    # Entropy calculation test
    assert data["mean_entropy"] > 0.0


def test_attention_causal_mask_detection():
    # 4-token causal self-attention
    causal_matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.6, 0.4, 0.0, 0.0],
        [0.2, 0.3, 0.5, 0.0],
        [0.1, 0.2, 0.3, 0.4],
    ]
    tokens = ["The", "model", "generates", "tokens"]

    data, summary, notes = inspect_attention(causal_matrix, tokens=tokens)
    assert data["is_causal"] is True
    assert any("causal" in n.lower() for n in notes)


def test_attention_cross_attention_rectangular_shapes():
    # Cross-attention: 2 queries attend to 5 keys
    cross_matrix = [
        [0.1, 0.2, 0.4, 0.2, 0.1],
        [0.05, 0.05, 0.1, 0.3, 0.5],
    ]
    q_toks = ["Dec1", "Dec2"]
    k_toks = ["Enc1", "Enc2", "Enc3", "Enc4", "Enc5"]

    data, summary, notes = inspect_attention(cross_matrix, query_tokens=q_toks, key_tokens=k_toks)

    assert data["is_cross_attention"] is True
    assert data["query_len"] == 2
    assert data["key_len"] == 5
    assert data["query_tokens"] == q_toks
    assert data["key_tokens"] == k_toks
    assert data["is_causal"] is False  # Rectangular matrix cannot be causal


def test_attention_multihead_individual_selection_and_average():
    # Multi-head attention: 2 heads, 3x3 tokens
    head0 = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    head1 = [
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]
    heads_tensor = torch.tensor([head0, head1])

    # 1. Inspect Head 0 specifically
    data_h0, _, _ = inspect_attention(heads_tensor, head=0)
    assert data_h0["active_head"] == "Head 0"
    assert data_h0["matrix"] == head0

    # 2. Inspect Head 1 specifically
    data_h1, _, _ = inspect_attention(heads_tensor, head=1)
    assert data_h1["active_head"] == "Head 1"
    assert data_h1["matrix"] == head1

    # 3. Default without head parameter -> Averaged across heads
    data_avg, _, _ = inspect_attention(heads_tensor, head=None)
    assert "Average" in data_avg["active_head"]
    # Check average of cell (0, 0): (1.0 + 0.0)/2 = 0.5
    assert abs(data_avg["matrix"][0][0] - 0.5) < 1e-4


def test_attention_head_comparison():
    head0 = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    head1 = [
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    all_heads = [head0, head1]

    cmp = compare_attention_heads(all_heads, 0, 1, ["Q0", "Q1"], ["K0", "K1"], False)

    assert cmp["available"] is True
    # Difference matrix: Head 1 - Head 0
    # (0, 0): 0.5 - 1.0 = -0.5
    # (0, 1): 0.5 - 0.0 = 0.5
    assert abs(cmp["diff_matrix"][0][0] - (-0.5)) < 1e-4
    assert abs(cmp["diff_matrix"][0][1] - 0.5) < 1e-4
    assert cmp["mae"] == 0.5


def test_attention_averaged_disables_individual_head_comparison():
    # Single head / averaged input
    single_head = [[[0.5, 0.5], [0.5, 0.5]]]
    cmp = compare_attention_heads(single_head, 0, 0, ["Q0", "Q1"], ["K0", "K1"], True)
    assert cmp["available"] is False
    assert "unavailable" in cmp["reason"].lower()


def test_attention_unnormalized_weights_not_silently_softmaxed():
    """Verify that unnormalized logits (or containing negatives) are preserved without silent softmax."""
    logits = [
        [2.5, -1.2, 0.0],
        [0.0, 3.1, -0.5],
    ]
    data, summary, notes = inspect_attention(logits)

    assert data["has_negatives"] is True
    assert data["is_softmax"] is False
    assert any("unnormalized" in n.lower() or "negative" in n.lower() for n in notes)

    # Values must match raw input exactly (not exponentiated or normalized)
    assert data["matrix"][0][0] == 2.5
    assert data["matrix"][0][1] == -1.2


def test_attention_torch_mha_module_capture():
    """Verify MultiheadAttention module capture without changing its state."""
    mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
    q = torch.randn(1, 4, 8)

    # In PyTorch, MHA with need_weights=True returns (output, weights)
    # Using sd.attention with MHA module:
    view = sd.attention(mha, input=q)

    assert view.kind == "attention"
    assert view.data["query_len"] == 4
    assert view.data["key_len"] == 4
    assert len(view.data["matrix"]) == 4


def test_attention_renderers_and_export():
    weights = [
        [0.7, 0.3],
        [0.2, 0.8],
    ]
    tokens = ["Tok1", "Tok2"]
    view = sd.attention(weights, tokens=tokens, backend="svg")

    # SVG export
    svg_str = view.to_figure("svg")
    assert "<svg" in svg_str
    assert "TRANSFORMER ATTENTION" in svg_str

    # Plotly export
    fig = view.to_figure("plotly")
    assert fig is not None

    # Matplotlib export
    mpl_fig = view.to_figure("matplotlib")
    assert mpl_fig is not None

    # HTML export
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        out_path = view.save(tmp.name)
        content = Path(out_path).read_text(encoding="utf-8")
        assert "SeeDL · Attention Intelligence" in content
        assert "token-table" in content
        assert "Query Attention Profile" in content


def test_vit_patch_attention_spatial_reshape():
    """Verify Vision Transformer [CLS] attention vector reshapes into authentic 2D spatial grid."""
    # 1 CLS token + 4 image patches (2x2 grid)
    # CLS attends to patches: [CLS, P(0,0), P(0,1), P(1,0), P(1,1)]
    attn = [
        [0.2, 0.4, 0.1, 0.2, 0.1],  # CLS token attention
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ]
    view = sd.vit_attention(attn, patch_grid=(2, 2), from_token=0, has_cls=True)

    assert view.kind == "feature_maps"
    layer = view.data["layers"][0]
    assert layer["original_shape"] == (1, 1, 2, 2)
    grid = layer["channels"][0]["grid"]
    assert len(grid) == 2
    assert len(grid[0]) == 2
    assert grid[0][0] == 0.4  # P(0, 0)
    assert grid[0][1] == 0.1  # P(0, 1)
    assert grid[1][0] == 0.2  # P(1, 0)
    assert grid[1][1] == 0.1  # P(1, 1)


def test_attention_invalid_inputs_raise_seedl_error():
    """Verify appropriate SeeDLError is raised when dimensions are invalid or head is out of range."""
    with pytest.raises(SeeDLError, match="at least 2-dimensional"):
        sd.attention([1.0, 2.0])  # 1D tensor

    # Head out of range
    with pytest.raises(SeeDLError, match="out of range"):
        sd.attention([[[0.5, 0.5], [0.5, 0.5]]], head=5)


def test_attention_3d_tensor_disambiguation():
    """Verify explicit and automatic disambiguation of 3D attention tensors."""
    # 3D tensor: [2, 3, 3]
    t3d = torch.ones(2, 3, 3) / 3.0

    # 1. Explicitly treated as batched: [batch=2, seq_q=3, seq_k=3]
    v_batched = sd.attention(t3d, is_batched=True, batch_index=1)
    assert v_batched.data["is_averaged"] is True
    assert v_batched.data["num_heads"] == 1
    assert any("disambiguated 3d tensor as batched" in n.lower() for n in v_batched.notes)

    # 2. Explicitly treated as multi-head: [heads=2, seq_q=3, seq_k=3]
    v_heads = sd.attention(t3d, is_batched=False)
    assert v_heads.data["num_heads"] == 2
    assert any("disambiguated 3d tensor as multi-head" in n.lower() for n in v_heads.notes)


def test_attention_invalid_distribution_disables_entropy():
    """Verify that unnormalized logits or tensors with negatives report mean_entropy as None."""
    # Raw unnormalized logits
    logits = [
        [3.0, -1.0, 0.5],
        [0.0, 2.0, -2.0],
    ]
    data, summary, notes = inspect_attention(logits)
    assert data["has_negatives"] is True
    assert data["is_softmax"] is False
    assert data["mean_entropy"] is None

    # Render Plotly without error
    view = sd.attention(logits, backend="plotly")
    fig = view.to_figure("plotly")
    assert fig is not None

    # Export HTML without error
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        out_path = view.save(tmp.name)
        html_text = Path(out_path).read_text(encoding="utf-8")
        assert "N/A (unnormalized/logits)" in html_text


def test_attention_supplied_mask_handling():
    """Verify that 1D key padding mask and 2D attention mask are tracked and noted."""
    # 3x3 attention matrix
    mat = [
        [0.7, 0.3, 0.0],
        [0.2, 0.8, 0.0],
        [0.5, 0.5, 0.0],
    ]
    # Key padding mask: Key 2 is padding (True = masked)
    mask_1d = [False, False, True]
    data, summary, notes = inspect_attention(mat, mask=mask_1d)

    assert data["mask"] is not None
    assert len(data["mask"]) == 3
    assert len(data["mask"][0]) == 3
    assert data["mask"][0][2] is True
    assert any("supplied attention mask" in n.lower() for n in notes)


def test_attention_html_and_svg_export_escaping():
    """Verify that malicious token strings with HTML/script tags are safely escaped."""
    malicious_tokens = ["<b>Tok1</b>", "<script>alert(1)</script>", 'Tok & "Quote"']
    mat = [
        [0.5, 0.3, 0.2],
        [0.1, 0.8, 0.1],
        [0.3, 0.3, 0.4],
    ]
    view = sd.attention(mat, tokens=malicious_tokens, backend="svg")

    # SVG export escaping
    svg_str = view.to_figure("svg")
    assert "<script>alert" not in svg_str
    assert "&lt;script&gt;alert" in svg_str

    # HTML export escaping
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        out_path = view.save(tmp.name)
        html_text = Path(out_path).read_text(encoding="utf-8")
        assert "</script><script>alert" not in html_text
        assert "escapeHtml" in html_text
