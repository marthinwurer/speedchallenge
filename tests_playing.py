import unittest

import glob

from load_dataset import load_speeds
from build_dataset import build_splits


class Tests(unittest.TestCase):
    def test_splitting(self):
        # practice splitting for cross validation
        splits = 5
        files = sorted(list(glob.glob("./data/train/*.png")))
        build_splits(files, splits, "frames")
        speeds = load_speeds("./data/train.txt")
        build_splits(speeds, splits, "speeds")


    # def test_load_splits(self):


if __name__ == '__main__':
    unittest.main()