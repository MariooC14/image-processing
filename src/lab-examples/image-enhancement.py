# Image Enhancement with Sobel, LoG, Laplacian filters
# Taken from https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html
import sys
import cv2 as cv
from cv2.typing import MatLike
from matplotlib import pyplot as plt
from src.utils import get_image
import numpy as np


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: show_image.py <filename>")

    img_path = sys.argv[1]
    img = get_image(img_path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    sobel_log_laplacian_enhancement_example(img)
    laplacian_of_gaussian_example(img)
    cv_16s_edge_detection_example(img)


def sobel_log_laplacian_enhancement_example(img: MatLike):
    laplacian = cv.Laplacian(img, cv.CV_64F)
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()


def laplacian_of_gaussian_example(img: MatLike):
    # Apply Gaussian Blur
    sigma = 1.0
    blurred_image = cv.GaussianBlur(img, (5, 5), sigma)

    # Apply Laplacian
    laplacian_img = cv.Laplacian(blurred_image, cv.CV_64F)
    laplacian_img = cv.convertScaleAbs(laplacian_img)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(laplacian_img, cmap='gray')
    plt.title('Laplacian of Gaussian (LoG)')
    plt.show()


def cv_16s_edge_detection_example(img):
    # Output dtype = cv.CV_8U
    sobelx8u = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=5)

    # Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
    sobelx64f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)

    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(sobelx8u, cmap='gray')
    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(sobel_8u, cmap='gray')
    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

    plt.show()

if __name__ == '__main__':
    main()