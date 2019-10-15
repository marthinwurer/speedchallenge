import torch

from load_dataset import SpeedDataset, load_speeds


def main():
    train_set = SpeedDataset("./data/frames/", load_speeds("./data/train.txt"))
    dataset_loader = torch.utils.data.DataLoader(train_set,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)

    # for i, data in enumerate(dataset_loader):
    #     print(data[1], data[2])
    #     if i > 3:
    #         break

    for i in range(3):
        data = train_set[i]
        print(data[1], data[2])

if __name__ == '__main__':
    main()