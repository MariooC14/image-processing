import os
from typing import Sequence

import cv2 as cv
import sys

in_dir = os.path.join(os.getcwd(), "..", "input-media")
out_dir = os.path.join(os.getcwd(), "..", "output-media")


def get_image(filename, flags: int):
    path = os.path.join(in_dir, filename)
    img = cv.imread(path, flags)

    if img is None:
        sys.exit("Could not read the image")

    return img


def write_image(filename, img):
    print("Writing image to output-media")
    cv.imwrite(os.path.join(out_dir, filename), img)


def show_image(img, dimension: Sequence[int]):
    img_resized = cv.resize(img, dimension)
    cv.imshow("Image", img_resized)
    cv.waitKey(0)


# Image must be grayscale
def get_image_histogram(img):
    histogram = [0 for _ in range(256)]
    height, width = img.shape

    for x in range(width):
        for y in range(height):
            pixel = img[y][x]
            histogram[pixel] += 1

    return histogram


def get_normalized_histogram(hist, num_pixels):
    normalized_hist = [0 for _ in range(256)]
    for i in range(256):
        normalized_hist[i] = hist[i] / num_pixels

    return normalized_hist


def running_sum(hist):
    c = [0 for _ in range(256)]
    c[0] = hist[0]
    for i in range(1, 256):
        c[i] = c[i-1] + hist[i]

    return c


def histogram_equalize(img):
    height, width = img.shape
    h = get_normalized_histogram(get_image_histogram(img), width*height)
    c = running_sum(h)

    out_img = img.copy()

    for x in range(width):
        for y in range(height):
            out_img[y][x] = round(255 * c[img[y][x]])

    return out_img