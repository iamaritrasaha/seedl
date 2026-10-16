"""Example: Trace a deep residual network using SeeDL's Neural Execution Tracer."""

from __future__ import annotations

import torch
import torch.nn as nn

import seedl as sd


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.init_bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.block1 = ResidualBlock(16)
        self.block2 = ResidualBlock(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.init_bn(self.init_conv(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def main():
    print("Tracing MiniResNet forward execution...")
    model = MiniResNet()
    example_input = torch.randn(1, 3, 32, 32)

    trace = sd.trace(model, input=example_input)
    print(f"Traced {len(trace.nodes)} operation nodes across model execution.")

    flow_view = sd.flow(trace)
    saved_path = flow_view.save("resnet_flow.html")
    print(f"Interactive Tensor Flow visualization saved to: {saved_path}")


if __name__ == "__main__":
    main()
