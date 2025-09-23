"""
V1: Histogram-Equalize a grayscale image
Put your input files inside the `input-media` folder and run this file with the file name.
"""
import sys
from utils import get_image
import cv2 as cv
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: show_image.py <filename>")

    filepath = sys.argv[1]

    img = get_image(filepath)

    gray_image = make_grayscale(img)
    show_image(gray_image)

    og_hist = get_image_histogram(gray_image)
    plot_histogram(og_hist)

    equalized_img = histogram_equalize(gray_image)
    show_image(equalized_img)

    equalized_histo = get_image_histogram(equalized_img)
    plot_histogram(equalized_histo, "Equalized Histogram")


def show_image(img):
    img_resized = cv.resize(img, (600, 400))
    cv.imshow("Image", img_resized)

    cv.waitKey(0)


def make_grayscale(img):
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray_image


# Image must be grayscale
def get_image_histogram(img):
    histogram = [0 for _ in range(256)]
    height, width = img.shape

    for x in range(width):
        for y in range(height):
            pixel = img[y][x]
            histogram[pixel] += 1

    return histogram


def plot_histogram(hist, title="Image Histogram"):
    plt.bar(range(256), hist)
    plt.title(title)
    plt.xlabel("Grayscale values")
    plt.ylabel("Pixel Frequency")
    plt.show()


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


if __name__ == "__main__":
    main()
