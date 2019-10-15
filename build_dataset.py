"""
Takes the comma.ai speed challenge dataset and breaks it up into frames.
"""
import glob

import cv2
from tqdm import tqdm

from load_dataset import load_speeds


def build_splits(items, splits, split_name):
    num_items = len(items)
    per_split = num_items // splits

    rest = items
    for i in range(splits):
        split = rest[:per_split]
        rest = rest[per_split:]

        with open("./data/split%s_%s.txt" % (i, split_name), "w") as f:
            for item in split:
                print(item, file=f)


def split_train(image_dir, label_file, splits=5):
    files = sorted(list(glob.glob(image_dir + "/*.png")))
    build_splits(files, splits, "frames")
    speeds = load_speeds(label_file)
    build_splits(speeds, splits, "speeds")


def image_generator(video):
    success = True
    while success:
        success, image = video.read()
        if success:
            yield image


def extract_frames(source, dest):
    vidcap = cv2.VideoCapture(source)

    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    print(width, height)

    # I want to clip from 32 to 32 + 320 in the y axis

    for count, frame in enumerate(tqdm(image_generator(vidcap))):
        # clip image
        frame = frame[32:(32+320),:]
        frame = cv2.resize(frame, (128, 64))
        cv2.imwrite(dest + "/frame%05d.png" % count, frame)


def main():
    extract_frames('./data/train.mp4', './data/train/')
    extract_frames('./data/test.mp4', './data/test/')
    split_train("./data/train", "./data/train.txt")


if __name__ == '__main__':
    main()


