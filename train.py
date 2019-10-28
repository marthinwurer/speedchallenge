import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from checkpoints import Checkpoint
from hyper import Hyperparameters, get_common_items
from load_dataset import get_selected_datasets, iter_net_transform
from network import SpeedNet
from test import test_net


@dataclass
class Statistics(object):
    loss: float = 0.0
    test_loss: float = 0.0
    training_steps: int = 0


def main():
    hyper = Hyperparameters()
    STACKS = hyper.STACKS
    LEARNING_RATE = hyper.LEARNING_RATE
    BATCH_SIZE = hyper.BATCH_SIZE
    EPOCHS = hyper.EPOCHS
    train_set = get_selected_datasets([0, 1, 2, 3], stacks=STACKS)
    dataset_loader = torch.utils.data.DataLoader(train_set,
                                                 batch_size=BATCH_SIZE, shuffle=True,
                                                 num_workers=4)

    test_set = get_selected_datasets([4], stacks=STACKS)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=4)

    net, device, optimizer, criterion = get_common_items(hyper)

    # test_loss = test_net(net, test_loader, device, criterion)
    # print("Starting Test loss: %s" % (test_loss,))

    training_steps = 0

    for epoch in range(EPOCHS):
        print("Starting epoch %s" % (epoch,))
        epoch_loss = 0.0
        with tqdm(iter(dataset_loader)) as t:
            for images, labels, frames in iter_net_transform(t, device):
                # Train
                optimizer.zero_grad()
                output = net(images)
                target = labels.view_as(output)  # make it the same shape as output
                loss = criterion(output, target)
                loss.backward()
                # gradient clip
                clip = .5
                for p in net.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-clip, clip)
                optimizer.step()

                epoch_loss += loss.item()
                t.set_description("Loss: %s" % (loss.item(),))
                # exit()

        # test the epoch
        print("Testing epoch %s" % (epoch,))
        test_loss = test_net(net, test_loader, device, criterion)
        print("Test loss: %s" % (test_loss,))

        # finish epoch and make checkpoint
        print("Finishing epoch %s" % (epoch,))
        average_loss = epoch_loss / len(dataset_loader)
        training_steps += len(dataset_loader)
        print("Training Loss: %s" % (average_loss,))
        stats = Statistics(loss=average_loss, test_loss=test_loss, training_steps=training_steps)
        # exit()
        checkpoint = Checkpoint(net, optimizer, training_steps=epoch, statistics=dataclasses.asdict(stats))
        checkpoint.save_state("./data/nets/check_%s.tor" % (epoch,))

    test_loss = test_net(net, test_loader, device, criterion)
    print("Final Test loss: %s" % (test_loss,))





if __name__ == '__main__':
    main()