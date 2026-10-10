"""SeeDL Activation Capture and Gradient Flow Diagnostics Example."""

import torch
import torch.nn as nn

import seedl as sd


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 16 * 16, 2)

    def forward(self, x):
        h = self.relu(self.conv(x))
        return self.fc(h.flatten(1))


model = ConvNet()
x = torch.randn(1, 3, 16, 16)

# 1. Capture intermediate layer activations & conv feature maps
print("--- Activations & Feature Maps ---")
act_view = sd.activations(model, input=x, backend="svg")
print(act_view.summary)

# 2. Inspect gradients without modifying parameters
print("\n--- Gradient Flow Inspection ---")
output = model(x).sum()
output.backward()

grad_view = sd.gradients(model, backend="svg")
print(grad_view.summary)
for note in grad_view.notes:
    print(f"  Note: {note}")
