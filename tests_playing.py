import unittest

import glob
from pprint import pprint

from load_dataset import load_speeds, get_speed_dataset, SpeedSplit, get_selected_datasets
from build_dataset import build_splits, split_train


class Tests(unittest.TestCase):
    def test_splitting(self):
        # practice splitting for cross validation
        # splits = 5
        # files = sorted(list(glob.glob("./data/train/*.png")))
        # build_splits(files, splits, "frames")
        # speeds = load_speeds("./data/train.txt")
        # build_splits(speeds, splits, "speeds")
        split_train("./data/train", "./data/train.txt")

    def test_load_single_split(self):
        dataset = SpeedSplit("./data", 0, 4)
        sample = dataset[1234]
        pprint(sample)

    def test_load_splits(self):
        dataset = get_speed_dataset()

        sample = dataset[12345]
        pprint(sample)

    def test_load_selected(self):
        dataset = get_selected_datasets([0,1,2,3])

        sample = dataset[12345]
        pprint(sample)


if __name__ == '__main__':
    unittest.main()