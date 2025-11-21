import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
from time import sleep
from scipy.signal import sepfir2d
from IPython.display import clear_output

from src.utils import get_image


def main():
    # compute space-time derivatives
    img = get_image("apple.jpg", flags=cv.IMREAD_GRAYSCALE)
    p = [0.030320, 0.249724, 0.439911, 0.249724, 0.030320]
    d = [-0.104550, -0.292315, 0.0, 0.292315, 0.104550]

    img_x = sepfir2d(img, d, p)  # spatial (x) derivative
    img_y = sepfir2d(img, p, d)  # spatial (y) deivative

    for th in range(0, 361, 5):
        plt.figure(figsize=(10, 10))
        plt.imshow(np.cos(np.radians(th)) * img_x + np.sin(np.radians(th)) * img_y, cmap='gray')
        plt.title(str(th))
        plt.show()
        sleep(0.1)
        clear_output(wait=True)


if __name__ == '__main__':
    main()