
import torch
import torch.nn as nn
import torch.nn.functional as F
from coordconv import CoordConv2d


class Flatten(nn.Module):
    """ From https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983/2"""
    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), -1)


class Sublayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.ModuleList([
            CoordConv2d(in_channels, out_channels, 3, padding=1),
            nn.Dropout(),
            nn.ReLU(),
            # nn.SELU(),
            # nn.BatchNorm2d(out_channels),
        ])

    def forward(self, input: torch.Tensor):
        x = input
        for layer in self.layers:
            x = layer(x)
        return x


class SpeedNet(nn.Module):
    def __init__(self, stacks):
        super().__init__()

        in_size = stacks * 3

        self.layers = nn.ModuleList([
            Sublayer(in_size, 128),  # from RGB
            Sublayer(128, 128),
            nn.MaxPool2d(2),  # now 64x32
            Sublayer(128, 128),
            Sublayer(128, 128),
            nn.MaxPool2d(2),  # now 32x16
            Sublayer(128, 64),
            Sublayer(64, 64),
            nn.MaxPool2d(2),  # now 16x8
            Sublayer(64, 64),
            Sublayer(64, 64),
            nn.MaxPool2d(2),  # now 8x4
            Sublayer(64, 64),
            Sublayer(64, 64),
            nn.AdaptiveMaxPool2d((1, 1)),  # global average pooling
            Flatten(),
        ])
        self.final = nn.Linear(64, 1)

    def forward(self, input: torch.Tensor):
        x = input
        for layer in self.layers:
            x = layer(x)

        x = self.final(x)

        x = x * 20  # speed averages about 18

        return x


class ResNetProjectionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, input: torch.Tensor):
        x = self.layer(input)
        return x


class ResNetPreactivationLayer(nn.Module):
    """
    From https://arxiv.org/abs/1603.05027 with dropout added from wide resnets (https://arxiv.org/abs/1605.07146)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            CoordConv2d(in_channels, out_channels, 3, padding=1),
            nn.Dropout(),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            CoordConv2d(in_channels, out_channels, 3, padding=1),
        ])

    def forward(self, input: torch.Tensor):
        x = input
        for layer in self.layers:
            x = layer(x)
        return input + x


class StackedPool(nn.Module):
    def __init__(self, pool):
        super().__init__()
        self.max = nn.MaxPool2d(pool)
        self.average = nn.AvgPool2d(pool)

    def forward(self, input: torch.Tensor):
        max_pool = self.max(input)
        avg_pool = self.average(input)
        catted = torch.cat([max_pool, avg_pool], dim=1)  # dim needs to be 1 because of batch
        return catted


class ResSpeedNet(nn.Module):
    def __init__(self, stacks):
        super().__init__()

        in_size = stacks * 3

        self.layers = nn.ModuleList([
            CoordConv2d(in_size, 64, 3),  # from RGB
            nn.ReLU(),
            ResNetPreactivationLayer(64, 64),
            StackedPool(2),  # now 64x32
            ResNetPreactivationLayer(128, 128),
            ResNetPreactivationLayer(128, 128),
            StackedPool(2),  # now 32x16
            ResNetPreactivationLayer(256, 256),
            ResNetPreactivationLayer(256, 256),
            StackedPool(2),  # now 16x8
            ResNetProjectionLayer(512, 256),
            ResNetPreactivationLayer(256, 256),
            ResNetPreactivationLayer(256, 256),
            StackedPool(2),  # now 8x4
            ResNetProjectionLayer(512, 256),
            ResNetPreactivationLayer(256, 256),
            ResNetPreactivationLayer(256, 256),
            nn.AdaptiveMaxPool2d((1, 1)),  # global average pooling
            Flatten(),
        ])
        self.final = nn.Linear(256, 1)

    def forward(self, input: torch.Tensor):
        x = input
        for layer in self.layers:
            x = layer(x)

        x = self.final(x)

        x = x * 12  # speed averages about 12

        return x

