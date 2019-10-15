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


def load_speeds(speed_file):
    with open(speed_file, 'r') as f:
        return [float(speed) for speed in f]
