import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class SpeedDataset(Dataset):

    def __init__(self, frames_dir, speeds, num_frames=16, size=(480, 640)):
        self.frames_dir = frames_dir
        self.speeds = speeds
        self.num_frames = num_frames
        self.size = size

    def __len__(self):
        return len(self.speeds)

    def __getitem__(self, item):
        """
        Get the n frames up to and including the last item.
        """
        frames = []
        speeds = []
        frame_nums = []
        for i in range(self.num_frames):
            frame_num = item - self.num_frames + i + 1
            try:
                speed = self.speeds[frame_num]
                frame = np.asarray(Image.open(self.frames_dir + "/frame%05d.png" % frame_num))
            except (IndexError, FileNotFoundError):
                frame = np.zeros((*self.size, 3))
                speed = 0

            frames.append(np.moveaxis(frame, -1, 0))  # make sure we're in channels first
            speeds.append(speed)
            frame_nums.append(frame_num)

        # stack frames and speeds, then return them both
        speeds = np.asarray(speeds)
        frames = np.stack(frames)
        frame_nums = np.asarray(frame_nums)

        return frames, speeds, frame_nums


def load_lines(file):
    with open(file, 'r') as f:
        return [line.rstrip() for line in f]


def load_speeds(speed_file):
    with open(speed_file, 'r') as f:
        return [float(speed) for speed in f]


class SpeedSplit(Dataset):
    def __init__(self, path, split, stacks, default_size=(64, 128)):
        # load the data files for the splits
        self.frames = load_lines(path + "/split%s_frames.txt" % split)
        self.speeds = load_speeds(path + "/split%s_speeds.txt" % split)
        self.stacks = stacks
        self.default_size = default_size


    def __len__(self):
        return len(self.speeds)

    def __getitem__(self, item):
        """
        Get the n frames up to and including the last item.
        """
        frames = []
        speeds = []
        frame_nums = []
        for i in range(self.stacks):
            frame_num = item - self.stacks + i + 1
            try:
                speed = self.speeds[frame_num]
                frame = np.asarray(Image.open(self.frames[frame_num]))
            except (IndexError, FileNotFoundError):
                frame = np.zeros((*self.default_size, 3))
                speed = 0

            frames.append(np.moveaxis(frame, -1, 0))  # make sure we're in channels first
            speeds.append(speed)
            frame_nums.append(frame_num)

        # stack frames and speeds, then return them both
        speeds = np.asarray(speeds)
        frames = np.stack(frames)
        frame_nums = np.asarray(frame_nums)

        return frames, speeds, frame_nums


class SplitSet(Dataset):
    def __init__(self, sets):
        self.sets = sets
        self.sizes = [len(dataset) for dataset in sets]
        self.size = sum(self.sizes)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        test_val = item
        for size, dataset in zip(self.sizes, self.sets):
            if test_val < size:
                return dataset[test_val]
            test_val -= size

        raise IndexError("Item not found: %s" % (item,))


def get_speed_dataset(stacks=4):
    sets = []
    for i in range(5):
        dataset = SpeedSplit("./data", i, stacks)
        sets.append(dataset)

    return SplitSet(sets)



