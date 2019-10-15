
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    """ From https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983/2"""
    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), -1)

class SpeedNet(nn.Module):
    def __init__(self, stacks):
        super().__init__()

        in_size = stacks * 3

        self.layers = [
            nn.Conv2d(in_size, 16, 3, padding=1),  # from RGB
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
            nn.AdaptiveMaxPool2d((1,1)),  # global average pooling
            Flatten(),
        ]
        self.final = nn.Linear(64, 1)

    def forward(self, input: torch.Tensor):
        x = input
        for layer in self.layers:
            x = F.relu(layer(x))

        x = self.final(x)

        return x



