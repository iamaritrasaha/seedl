"""SeeDL Model Architecture and Training Observability Example."""

import torch
import torch.nn as nn

import seedl as sd


# 1. Define a PyTorch Neural Network
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
        h = torch.relu(self.conv1(x))
        p = self.pool(h).flatten(1)
        return self.fc(p)


model = Classifier()
sample_input = torch.randn(2, 3, 32, 32)

# 2. Inspect Model Architecture and Shape Propagation
print("--- Model Inspection ---")
model_view = sd.see(model, input=sample_input, backend="svg")
print(model_view.summary)

# 3. Track Training Metrics with Tracker
print("\n--- Training Observability ---")
tracker = sd.watch(name="run_resnet_head")
tracker.log(epoch=1, loss=0.85, val_loss=0.90, accuracy=0.65)
tracker.log(epoch=2, loss=0.55, val_loss=0.62, accuracy=0.78)
tracker.log(epoch=3, loss=0.35, val_loss=0.48, accuracy=0.89)

train_view = sd.see(tracker, backend="svg")
print(train_view.summary)

# 4. Compare Two Training Runs
hist_a = {"epochs": [1, 2, 3], "loss": [0.9, 0.6, 0.4]}
hist_b = {"epochs": [1, 2, 3], "loss": [0.8, 0.5, 0.2]}
comp_view = sd.compare(hist_a, hist_b, names=("Adam", "AdamW"), backend="svg")
print(f"Comparison: {comp_view.summary}")
