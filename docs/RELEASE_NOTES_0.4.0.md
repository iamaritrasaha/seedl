# SeeDL 0.4.0 — Vision & Attention Intelligence

**Status:** Release candidate source prepared for review; not yet published to PyPI. Version metadata is 0.4.0. Publication, tagging, and deployment remain separate manual actions.

---

## Overview

SeeDL 0.4.0 introduces **Vision & Attention Intelligence**, expanding SeeDL into an interactive visual analysis environment for Convolutional Neural Network (CNN) feature maps, Transformer multi-head attention routing, and intermediate neural representations.

All capabilities are designed around scientific correctness: inspecting authentic activations captured from genuine PyTorch models or user-supplied tensors without synthetic demonstrations, silent conversions, or invented data.

---

## CNN Feature-Map Explorer

Inspect intermediate convolutional representations across channels, layers, and spatial coordinates:

- **Model & Layer Support**: Captures genuine forward activations from `torch.nn.Conv2d`, nested `torch.nn.Sequential` pipelines, custom `torch.nn.Module` architectures, activation layers (`ReLU`, `GELU`, `SiLU`, `Tanh`, `Sigmoid`, `LeakyReLU`), and pooling layers (`MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`).
- **Repeated Module Invocations**: Distinctly captures multiple forward calls of shared or reused modules in recurrent or multi-branch pipelines without collision.
- **Interactive Inspection**: Layer selector, channel selector, multi-channel thumbnail gallery, and spatial hover inspection displaying exact coordinate `(y, x)` and activation values.
- **Channel Diagnostics**: Computes min, max, mean, standard deviation, zero/sparsity fraction, and positive fraction per channel and globally across layers.
- **Signed Activation Normalization**: Three normalization modes:
  - `per_channel`: Normalized to the selected channel's minimum and maximum.
  - `layer_global`: Normalized to the overall layer's global minimum and maximum.
  - `zero_centered`: Symmetrical range $[-M, +M]$ around zero using diverging color palettes for signed activations.
- **Singleton Spatial Handling**: Robust handling of singleton spatial dimensions (`[B, C, 1, 256]`, `[B, C, 256, 1]`, `[B, C, 1, 1]`) without dimensionality collapse or canvas distortion.
- **Module State Integrity**: Accurately preserves the original `training` flag for every module (including child modules in mixed `eval()`/`train()` configurations). Hooks are cleanly detached in `finally` blocks, and model buffers are safeguarded.

---

## Transformer Attention Intelligence

Inspect attention probability distributions, query-key alignment, multi-head routing, and sequence interaction patterns:

- **Attention Matrix Heatmaps**: Query tokens along rows ($Q_i$) and key tokens along columns ($K_j$), visualizing empirical routing probabilities $P(K_j \mid Q_i)$.
- **Multi-Head Routing**: Inspect individual attention heads (Head 0, Head 1, ...) or composite averaged attention across all heads.
- **3D Tensor Shape Disambiguation**: Explicit `is_batched` flag and dimension checking distinguishes `[heads, seq_q, seq_k]` from `[batch, seq_q, seq_k]` without ambiguous guessing.
- **MultiheadAttention Transparency**: Handles PyTorch's `nn.MultiheadAttention` honestly, clarifying when PyTorch averages attention across heads by default (`need_weights=True`, `average_attn_weights=True`) so averaged weights are never mislabeled as individual heads.
- **Distribution Validity & Entropy**: Computes Shannon entropy $H(Q_i) = -\sum_j p_{ij} \log_2 p_{ij}$ strictly on valid probability distributions. Raw logits or negative weights are presented faithfully without silent softmax manipulation.
- **Causal Mask Detection**: Automatically identifies lower-triangular causal attention patterns.
- **Cross-Attention Support**: Supports unequal query and key dimensions ($seq_q \neq seq_k$) with separate query and key token labels.

---

## Multi-Head Attention Comparisons

Directly compare routing patterns between different attention heads:

- Pairwise comparison between any two heads ($Head_A \text{ vs } Head_B$).
- Difference heatmap visualizing elementwise shifts in query-key routing.
- Quantitative similarity metrics: Frobenius norm of difference, Mean Absolute Difference (MAE), and Cosine Similarity.
- Graceful degradation: Head comparison is cleanly disabled with explanatory diagnostics when individual heads are unavailable (e.g. averaged attention).

---

## Token-Level Exploration

Interactive token navigation revealing fine-grained attention mechanics:

- **Query Token Focus**: Selecting a query token displays a dynamic 1D bar chart of its attention distribution across all key tokens.
- **Key Token Influx**: Measures aggregate incoming attention across query tokens to reveal syntactic or semantic attractor tokens.
- **Security & Escaping**: All user-supplied token labels are securely escaped in HTML exports, preventing XSS injection or document structure corruption.

---

## Vision Transformer (ViT) Patch Attention

Bridge token attention back to spatial 2D coordinates:

- `sd.vit_attention()` reshapes 1D token attention vectors (such as `[CLS]` token attention) into authentic 2D spatial patch heatmaps over the original image patch grid.
- Correct handling of patch dimensions, grid layouts, and `has_cls` token offsets without inventing spatial coordinates.

---

## Aligned CNN Feature Comparisons

Compare internal representations across layers, inputs, or training checkpoints:

- Elementwise spatial difference maps $(B - A)$, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and maximum difference.
- Strict spatial alignment: Computes difference heatmaps only when spatial dimensions match. If dimensions differ, SeeDL displays side-by-side maps with an explicit shape-mismatch notice rather than subtracting unrelated coordinates.

---

## Offline Self-Contained HTML Exports

Export standalone, dependency-free interactive HTML files for sharing and reporting:

- Interactive client-side UI with tabs, layer/channel selectors, hover tooltips, and normalization mode toggles.
- Zero external CDN dependencies; functions entirely offline.
- Built-in SVG and Canvas renderers alongside optional Plotly and Matplotlib backends.

---

## Public APIs

```python
import seedl as sd
import torch
import torch.nn as nn

# 1. CNN Feature Maps
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
)
img = torch.randn(1, 3, 32, 32)
view = sd.features(model, input=img)
view.save("cnn_features.html")

# 2. Transformer Attention
attn_weights = torch.softmax(torch.randn(4, 8, 8), dim=-1)
tokens = ["The", "neural", "network", "attends", "to", "these", "input", "tokens"]
view = sd.attention(attn_weights, tokens=tokens)
view.save("attention.html")

# 3. Vision Transformer Patch Attention
vit_attn = torch.softmax(torch.randn(197, 197), dim=-1)
view = sd.vit_attention(vit_attn, patch_grid=(14, 14), from_token=0, has_cls=True)
view.save("vit_patch.html")

# 4. Feature Comparison
diff_view = sd.compare(view_a, view_b, names=("Baseline", "Perturbed"))
diff_view.save("feature_comparison.html")
```

---

## Performance, Memory, and Scientific Limitations

- **Observational Attention**: Attention visualizations show observational routing weights ($P(K_j \mid Q_i)$). They do not represent causal explanations of model predictions or proofs of feature importance.
- **Bounded Retention**: Configurable resource budgets protect memory:
  - `max_channels` (default 64)
  - `max_spatial_dim` (default 128; downsampled via adaptive average pooling when exceeded)
  - `max_seq_len` (default 256)
  - `max_heads` (default 32)
- **Memory Scope**: Retention limits apply to stored visualization payloads on CPU. They are not process-wide or GPU memory limits. Peak GPU allocations during PyTorch model forward passes depend on the model and input size.
- **Model Buffer Safety**: Module tracking bounds buffer cloning to small tracking buffers (`numel <= 8192`) to prevent GPU/CPU allocation spikes.
- **Receptive Fields**: Feature map coordinates represent activation tensor indices, not theoretical receptive fields back to pixel space.

---

## Compatibility

- Requires **Python 3.10+**.
- **Zero mandatory dependencies**: Core functionality requires no third-party packages. `torch`, `numpy`, `plotly`, `matplotlib`, and `qiskit` remain optional extras.
- Full backward compatibility: All SeeDL 0.3.0 training playback (`sd.watch`), quantum gate evolution (`sd.circuit`), MLP 3D exploration, metric tracking, and export APIs remain fully supported.

---

## Verification & Testing Results

- **Full Test Suite**: **373 passed, 1 skipped** on Python 3.13 (1 accelerator-only test skipped because CUDA/ROCm is not available in the CPU test environment; no GPU testing is claimed).
- **Multi-Version Wheel Compatibility**: Clean wheel installations passed core import, version verification, SVG rendering, and zero-mandatory-dependency checks on **Python 3.10, 3.11, 3.12, and 3.13**.
- **Source Distribution**: A clean sdist installation passed all core zero-dependency checks on **Python 3.13**.
- **Code Quality & Packaging**: Ruff lint/format, mypy type checks, `uv lock --check`, `twine check --strict`, and `scripts/verify_release.py --tag v0.4.0 --check-dist` passed cleanly.
- **Website Compatibility**: Verified isolated build and provenance checks (`generate.py` and `check.py`) using published `seedl==0.3.0`.
- **Status**: Release candidate prepared locally for review; unpushed, untagged, and not yet published to PyPI.
