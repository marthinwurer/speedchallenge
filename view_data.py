import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from checkpoints import Checkpoint
from load_dataset import get_selected_datasets
from hyper import Hyperparameters, get_common_items


# def un_torchify(images):
#     output = []
#     for image in images:
#

def main():
    hyper = Hyperparameters()
    STACKS = hyper.STACKS
    LEARNING_RATE = hyper.LEARNING_RATE
    # BATCH_SIZE = hyper.BATCH_SIZE
    BATCH_SIZE = 4
    test_set = get_selected_datasets([4], stacks=STACKS)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=4)

    dataiter = iter(test_loader)
    images, labels, frames = dataiter.next()

    print("Shape: %s" % (images[0].shape,))

    plt.figure(figsize=(9, 3))
    plt.imshow(images[0, 0:3])
    plt.subplot(132)
    plt.imshow(images[1, 0:3])
    plt.subplot(133)
    plt.imshow(images[2, 0:3])
    plt.suptitle('Test Data')
    plt.show()


if __name__ == '__main__':
    main()
