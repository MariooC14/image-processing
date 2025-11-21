# From https://docs.opencv.org/3.4/de/dbc/tutorial_py_fourier_transform.html
import sys

import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from matplotlib import pyplot as plt

from src.utils import get_image


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: show_image.py <filename>")

    filepath = sys.argv[1]
    img = get_image(filepath, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    fourier_transform_example(img)


def fourier_transform_example(img: MatLike):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)


if __name__ == '__main__':
    main()