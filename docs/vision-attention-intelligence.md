# SeeDL 0.4.0 — Vision & Attention Intelligence

Welcome to the **SeeDL 0.4.0 Development Preview**. This milestone expands SeeDL with interactive visual intelligence for Convolutional Neural Network (CNN) feature maps, Transformer multi-head attention routing, and intermediate neural representations.

---

## 1. Principles & Scientific Integrity

SeeDL visualizations are designed around rigorous scientific correctness:

- **Authentic Activations**: Visualizations display genuine activations and weights captured from real PyTorch models or user tensors. No synthetic demonstrations or fabricated patterns are presented as captured data.
- **Model Training State Preservation**: Model inspection runs in `eval()` mode under `torch.no_grad()`. Forward hooks are strictly removed after capture, buffers are restored, and `model.training` flags are preserved.
- **Observational Attention**: Attention visualizations show empirical query-key routing probabilities ($P(K_j \mid Q_i)$). They are explicitly labeled as observational attention weights, not causal explanations of model decisions or proofs of feature importance.
- **No Silent Conversions**: If unnormalized logits or negative weights are supplied, SeeDL displays them faithfully without silently applying softmax or modifying values.
- **Aligned Comparisons**: Difference heatmaps are calculated only when spatial dimensions match. Misaligned dimensions produce an explicit shape-mismatch diagnostic instead of subtracting unrelated coordinates.
- **Bounded Resource Budgets**: Memory and spatial limits (`max_channels`, `max_spatial_dim`, `max_seq_len`, `max_heads`) guard CPU/GPU allocations with clear metadata when downsampling occurs.
- **Zero Mandatory Dependencies**: SeeDL continues to function without mandatory dependencies, providing built-in SVG and pure offline interactive HTML fallbacks alongside optional Plotly and Matplotlib backends.

---

## 2. CNN Feature Map Explorer

The CNN Feature Map Explorer inspects intermediate convolutional representations across channels, layers, and spatial coordinates.

### Minimum Supported Models & Layers

- `torch.nn.Conv2d` and all standard 2D convolutional layers
- Nested `torch.nn.Sequential` and custom `torch.nn.Module` architectures
- Activation layers: `ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`
- Pooling layers: `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`
- Reused module calls (multiple forward invocations captured distinctly without overwriting)
- Common tensor shapes: `[B, C, H, W]` and `[C, H, W]`

### Quick Example

```python
import torch
import torch.nn as nn
import seedl as sd

# Define a convolutional model
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
)

# Run feature map inspection
sample_image = torch.randn(1, 3, 32, 32)
view = sd.features(model, input=sample_image)

# Or via sd.see()
view = sd.see(model, input=sample_image, view="features")

# Export offline self-contained HTML
view.save("cnn_features.html")
```

### Inspected Metrics & Capabilities

- **Layer Selector**: Switch between captured convolutional, pooling, and activation blocks.
- **Channel Selector & Thumbnail Gallery**: Browse individual channels or the entire channel gallery.
- **Spatial Coordinates**: Real-time hover inspection showing coordinate `(y, x)` and exact activation value.
- **Channel Diagnostics**: Min, Max, Mean, Standard Deviation, Zero/Sparsity fraction, Positive fraction.
- **Normalization Modes**:
  - `per_channel`: Scaled to the selected channel's min..max.
  - `layer_global`: Scaled to the overall layer's min..max across all channels.
  - `zero_centered`: Symmetrical range $[-M, +M]$ around zero using diverging colorscales for signed feature maps.

---

## 3. CNN Feature Comparison

Compare representations between different layers, different inputs, or model states before and after training:

```python
# Compare feature maps between two inputs
view_a = sd.features(model, input=input_a)
view_b = sd.features(model, input=input_b)

# Aligned spatial comparison
diff_view = sd.compare(view_a, view_b, names=("Input A", "Input B"))
diff_view.save("cnn_comparison.html")
```

If spatial dimensions $(H, W)$ match, SeeDL computes elementwise difference $(B - A)$, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and max difference. If dimensions differ (e.g. $14 \times 14$ vs $7 \times 7$), SeeDL displays side-by-side maps with an explicit shape mismatch warning, refusing to subtract unrelated coordinates.

---

## 4. Transformer Attention Explorer

Inspect transformer attention probability distributions, query-key alignment, multi-head routing, and causal patterns.

### Supported Input Formats

- Attention tensors: `[heads, seq_q, seq_k]`, `[batch, heads, seq_q, seq_k]`, or `[seq_q, seq_k]`
- PyTorch tensors, NumPy arrays, or Python nested lists
- `torch.nn.MultiheadAttention` module with `input=(query, key, value)`
- External transformer attention tensors (without requiring Hugging Face as a mandatory dependency)

### Quick Example

```python
import torch
import seedl as sd

# 4-head attention weights over a 6-token sentence
attention_weights = torch.softmax(torch.randn(4, 6, 6), dim=-1)
tokens = ["The", "quick", "brown", "fox", "jumps", "over"]

view = sd.attention(attention_weights, tokens=tokens)
view.save("transformer_attention.html")
```

### Visualizations & Features

1. **Attention Matrix Heatmap**: Rows represent Queries ($Q_i$), Columns represent Keys ($K_j$).
2. **Individual Head Selection**: Switch between Head 0, Head 1, ..., or view the Averaged attention across heads.
3. **Query Token Inspection**: Select any query token to inspect its 1D attention distribution over key tokens.
4. **Key Token Inspection**: Measure total incoming attention to specific key tokens.
5. **Entropy & Focus**: Shannon entropy $H(Q_i) = -\sum_j p_{ij} \log_2 p_{ij}$ measures attention concentration.
6. **Causal Mask Detection**: Automatically detects lower-triangular causal attention patterns.
7. **Cross-Attention Support**: Supports unequal query and key dimensions with explicit `query_tokens` and `key_tokens`.

---

## 5. Multi-Head Attention Comparison

Compare attention patterns between two attention heads:

```python
# Direct comparison between Head 0 and Head 2
view = sd.attention(attention_weights, tokens=tokens, compare_heads=(0, 2))
view.save("head_comparison.html")
```

Computes Frobenius norm of difference, Mean Absolute Difference (MAE), and Cosine Similarity between heads. For averaged attention tensors where individual heads were not exposed, head comparison is cleanly disabled with an explanatory note.

---

## 6. Vision Transformer (ViT) Patch Attention

Convert Vision Transformer token attention slices into authentic 2D spatial patch maps:

```python
# [CLS] token attending across a 14x14 patch grid
view = sd.vit_attention(vit_attention_weights, patch_grid=(14, 14), from_token=0, has_cls=True)
view.save("vit_patch_attention.html")
```

---

## 7. Memory & Performance Safeguards

To prevent memory exhaustion when working with deep models and long sequences:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `max_channels` | `64` | Maximum channels captured per layer |
| `max_spatial_dim` | `128` | Maximum spatial resolution before adaptive pooling downsampling |
| `max_seq_len` | `256` | Maximum query/key sequence tokens captured |
| `max_heads` | `32` | Maximum attention heads captured |
| `batch_index` | `0` | Selected batch sample from batched tensors |

All tensors are detached from autograd immediately and converted to CPU slices before rendering.

---

## 8. Exporting and Backends

- **Plotly**: Interactive figures with native dropdown menus, sliders, and subplots (`backend="plotly"`).
- **SVG**: High-resolution static vector charts (`backend="svg"`).
- **Matplotlib**: Publication-ready scientific figures (`backend="matplotlib"`).
- **Offline HTML**: Self-contained client-side HTML files with responsive controls and zero CDN requirements (`view.save("file.html")`).
