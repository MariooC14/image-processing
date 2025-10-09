# Image thresholding
# Taken from https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
import sys
import cv2 as cv
from cv2.typing import MatLike

from src.utils import get_image
from matplotlib import pyplot as plt

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: show_image.py <filename>")

    filepath = sys.argv[1]
    img = get_image(filepath, cv.IMREAD_GRAYSCALE)
    image_thresholding_example(img)
    image_adaptive_thresholding_example(img)


def image_thresholding_example(img: MatLike):
    ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
    ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
    ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']

    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


def image_adaptive_thresholding_example(img: MatLike):
    img = cv.medianBlur(img, 5)

    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']

    images = [img, th1, th2, th3]

    for i in range(4):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


# Note there is also Otsu's Binarization


if __name__ == '__main__':
    main()