"""
Fourier Transform
From https://docs.opencv.org/3.4/de/dbc/tutorial_py_fourier_transform.html
"""

import sys

from cv2.typing import MatLike

from src.utils import get_image
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: show_image.py <filename>")

    filepath = sys.argv[1]
    img = get_image(filepath, cv.IMREAD_GRAYSCALE)
    assert img is not None, "File could not be read, check with os.path.exist()"
    # fast_fourier_transform_example(img)
    inverse_transform_example(img)


def fast_fourier_transform_example(img: MatLike):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


def inverse_transform_example(img: MatLike):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Create a high-pass filter mask, you can also design low-pass filter or any filter of your choice
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow - 30:crow + 31, ccol - 30:ccol + 31] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(img_back, cmap='gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    main()