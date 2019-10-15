"""
Takes the comma.ai speed challenge dataset and breaks it up into frames.
"""


import cv2
from tqdm import tqdm


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


if __name__ == '__main__':
    main()