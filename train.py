import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from load_dataset import SpeedDataset, load_speeds, get_selected_datasets
from network import SpeedNet


def main():
    stacks = 4
    LEARNING_RATE = 0.001
    train_set = get_selected_datasets([0,1,2,3], stacks=stacks)
    dataset_loader = torch.utils.data.DataLoader(train_set,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)

    net = SpeedNet(stacks=stacks)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.MSELoss()
    last_loss = 999.999

    with tqdm(iter(dataset_loader)) as t:
        for images, labels, frames in t:
            images = images.to(device, dtype=torch.float)
            labels = labels[:,-1]
            labels = labels.to(device, dtype=torch.float)

            # preprocess image into 0..1
            images = images / 255.0

            # Train
            optimizer.zero_grad()
            output = net(images)
            target = labels.view_as(output)  # make it the same shape as output
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


            last_loss = loss
            t.set_description("Loss: %s" % (float(loss),))
            # exit()




if __name__ == '__main__':
    main()