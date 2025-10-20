import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def main():
    task1()
    # task2()


def task1():
    img = cv.imread("img_1.png", cv.IMREAD_GRAYSCALE)

    img_equalized = cv.equalizeHist(img)
    cv.imwrite("img_equalized.png", img_equalized)

    img_threshed = apply_thresholding(img_equalized)
    cv.imwrite("img_threshed.png", img_threshed)

    # Step 4: Skeletonization
    # Create empty skeleton
    skel = np.zeros(img_threshed.shape, np.uint8)
    # Cross Shaped Kernel
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    while True:
        # Open image
        open = cv.morphologyEx(img_threshed, cv.MORPH_OPEN, element)
        # Subtract open from the original image
        temp = cv.subtract(img_threshed, open)
        # Erode original image and refine the skeleton
        eroded = cv.erode(img_threshed, element)
        skel = cv.bitwise_or(skel, temp)
        img_threshed = eroded.copy()

        # If there are no white pixels left, i.e. the image has been completely eroded, exit the loop
        if cv.countNonZero(img_threshed) == 0:
            break

    cv.imwrite("img_skelly.png", skel)


def task2():
    img = cv.imread("img_2.png", cv.IMREAD_GRAYSCALE)
    sigma = 25

    rows, cols = img.shape
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    crow, ccol = rows // 2, cols // 2
    y = np.arange(rows) - crow
    x = np.arange(cols) - ccol
    X, Y = np.meshgrid(x, y)
    D2 = X ** 2 + Y ** 2

    H = np.exp(-D2 / (2 * (sigma ** 2)))

    fshift_filtered = fshift * H
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back = np.clip(img_back, 0, 255)
    img_back = img_back.astype(np.uint8)

    plt.subplot(121), plt.imshow(img_back, cmap='gray')
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


# Returns binary inverted image
def apply_thresholding(img):
    thresh_val = 100
    max_val = 255
    _, img_bin = cv.threshold(img, thresh_val, max_val, cv.THRESH_BINARY)
    _, img_bin_inv = cv.threshold(img, thresh_val, max_val, cv.THRESH_BINARY_INV)
    _, img_bin_trunc = cv.threshold(img, thresh_val, max_val, cv.THRESH_TRUNC)
    _, img_bin_tozero = cv.threshold(img, thresh_val, max_val, cv.THRESH_TOZERO)
    _, img_bin_tozero_inv = cv.threshold(img, thresh_val, max_val, cv.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']

    images = [img, img_bin, img_bin_inv, img_bin_trunc, img_bin_tozero, img_bin_tozero_inv]

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

    return img_bin_inv


if __name__ == '__main__':
    main()