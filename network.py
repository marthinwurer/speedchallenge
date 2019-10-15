
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeedNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = [
            nn.Conv2d(3, 16, 3, padding=1),  # from RGB
            nn.Conv2d(16, 16, 3, padding=1),
            nn.MaxPool2d(2),  # now 64x32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.MaxPool2d(2),  # now 32x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2),  # now 16x8
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.MaxPool2d(2),  # now 8x4
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
        ]

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = F.relu(layer(x))

        return x



