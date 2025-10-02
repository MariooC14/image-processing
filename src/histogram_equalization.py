"""
V1: Histogram-Equalize an image
Put your input files inside the `input-media` folder and run this file with the file name.
"""
import sys
import cv2 as cv
import numpy as np
from cv2.typing import MatLike
import matplotlib.pyplot as plt

from src.utils import get_image, show_image

DEFAULT_DIMENSION = np.array([600, 400])


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: show_image.py <filename>")

    filepath = sys.argv[1]
    image = get_image(filepath, cv.IMREAD_GRAYSCALE)

    histogram_equalization_with_numpy(image)
    histogram_equalization_with_opencv(image)


def histogram_equalization_with_numpy(img: MatLike):
    hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 256))

    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    print(cdf)
    print(cdf_normalized)

    plt.title("Cumulative Sum and Normalized Histogram of Image")
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), bins=256, range=(0, 256), color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf = np.ma.filled(cdf_masked, 0).astype('uint8')
    img2 = cdf[img]

    show_image(img2, DEFAULT_DIMENSION)


def histogram_equalization_with_opencv(img: MatLike):
    equ = cv.equalizeHist(img)
    res = np.hstack((img, equ)) #stacking images side-by-side
    show_image(res, DEFAULT_DIMENSION*2)


if __name__ == "__main__":
    main()