from dataclasses import dataclass

import torch
from torch import optim, nn

from network import SpeedNet, ResSpeedNet


@dataclass
class Hyperparameters(object):
    LEARNING_RATE = 0.001
    BATCH_SIZE = 128
    EPOCHS = 150
    STACKS = 16


def get_common_items(hyper):
    net = SpeedNet(stacks=hyper.STACKS)
    # net = ResSpeedNet(stacks=hyper.STACKS)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=hyper.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(net.parameters(), lr=hyper.LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()

    return net, device, optimizer, criterion

