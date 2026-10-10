"""SeeDL Explainability, Predictions, and Vector Embeddings Example."""

import torch
import torch.nn as nn

import seedl as sd

# 1. Prediction Ranking & Calibration
print("--- Prediction Inspection ---")
logits = [2.4, 0.2, -1.5, 0.9]
labels = ["cat", "dog", "bird", "fish"]
pred_view = sd.predictions(logits, labels=labels, task="classification", backend="svg")
print(pred_view.summary)

# 2. Explainability: Feature Attribution
print("\n--- Feature Attribution ---")
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
sample = torch.tensor([[1.0, -0.5, 2.0, 0.1]])

# Integrated Gradients satisfies path completeness
exp_view = sd.explain(model, input=sample, method="integrated_gradients", backend="svg")
print(exp_view.summary)

# 3. Vector Embeddings: Cosine Similarity and 2D PCA
print("\n--- Embeddings Exploration ---")
embeddings = [
    [1.0, 0.1, 0.0],
    [0.9, 0.2, 0.1],
    [0.0, 0.9, 0.8],
    [0.1, 0.8, 0.9],
]
emb_view = sd.embeddings(embeddings, labels=["doc1", "doc2", "query1", "query2"], backend="svg")
print(emb_view.summary)
