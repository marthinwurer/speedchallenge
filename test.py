import torch
from tqdm import tqdm

from checkpoints import Checkpoint
from load_dataset import get_selected_datasets, iter_net_transform
from hyper import Hyperparameters, get_common_items


class TestContext(object):
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.train(False)

    def __exit__(self, type, value, traceback):
        self.model.train(True)


def test_net(net, dataset_loader, device, criterion):
    epoch_loss = 0.0
    with torch.no_grad(), TestContext(net):
        with tqdm(iter(dataset_loader)) as t:
            for images, labels, frames in iter_net_transform(t, device):
                # Train
                output = net(images)
                target = labels.view_as(output)  # make it the same shape as output
                loss = criterion(output, target)

                epoch_loss += loss.item()
                t.set_description("Loss: %s" % (loss.item(),))

    average_loss = epoch_loss / len(dataset_loader)
    return average_loss



def main():
    hyper = Hyperparameters()
    STACKS = hyper.STACKS
    LEARNING_RATE = hyper.LEARNING_RATE
    BATCH_SIZE = hyper.BATCH_SIZE
    test_set = get_selected_datasets([4], stacks=STACKS)
    test_loader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=BATCH_SIZE, shuffle=True,
                                                 num_workers=4)

    net, device, optimizer, criterion = get_common_items(hyper)

    checkpoint = Checkpoint(net, optimizer)
    checkpoint.load_state("./data/nets/check_49.tor")
    net = checkpoint.model


    loss = test_net(net, test_loader, device, criterion)
    print(loss)


if __name__ == '__main__':
    main()