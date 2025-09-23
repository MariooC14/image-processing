import os
import cv2 as cv
import sys

in_dir = os.path.join(os.getcwd(), "..", "input-media")
out_dir = os.path.join(os.getcwd(), "..", "output-media")


def get_image(filename):
    path = os.path.join(in_dir, filename)
    img = cv.imread(path)

    if img is None:
        sys.exit("Could not read the image")

    return img


def write_image(filename, img):
    print("Writing image to output-media")
    cv.imwrite(os.path.join(out_dir, filename), img)