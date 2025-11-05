import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cv2.typing import MatLike

from src.utils import get_image


def main():
    image_names = ["apple.jpg", "flowers.jpg", "Lena.png"]

    for image_name in image_names:
        img = get_image(image_name, cv.IMREAD_COLOR_BGR)
        # task1(img)
        # task2(img)
        # task3(img)
        task4(img)


def task1(img: MatLike):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, ksize=(5, 5), sigmaX=1.0, sigmaY=1.0)

    edges1 = cv.Canny(blur, 50, 150)
    edges2 = cv.Canny(blur, 50, 200)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(edges1, cmap="gray")
    plt.title("Canny 50/150")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(edges2, cmap="gray")
    plt.title("Canny 100/200")
    plt.axis("off")
    plt.show()


"""
Detect corners using both the Harris corner detectoer (cv.cornerHarris) and the 
Shi-Tomasi (Tomasi-Kanade) method (cv.goodFeaturesToTrack).
Mark detected points on the original color images
"""
def task2(img: MatLike):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)
    # Harris
    harris = cv.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
    harris = cv.dilate(harris, None)
    harris_img = img.copy()
    harris_img[harris > 0.01 * harris.max()] = [255, 0, 0]  # mark in red

    # Shi–Tomasi (Tomasi–Kanade)
    shi_img = img.copy()

    pts = cv.goodFeaturesToTrack(gray, maxCorners=150, qualityLevel=0.01, minDistance=8)
    if pts is not None:
        pts = np.int32(pts)
        for p in pts:
            x, y = p.ravel()
            cv.circle(shi_img, (x, y), 3, (255, 0, 0), -1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(cv.cvtColor(harris_img, cv.COLOR_BGR2RGB))
    plt.title("Harris")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(shi_img, cv.COLOR_BGR2RGB))
    plt.title("Shi–Tomasi")
    plt.axis("off")
    plt.show()


"""
# Image Registration Using SIFT Features
1. Create a transformed version of one image (e.g. rotate the Lena image about 15º and slightly scale it.
2. Detect keypoints and descriptors in both images using the SIFT detector.
3. Match the descriptors using BFMatcher on FLANN and apply Lowe's ratio to test filter good matches
4. Estimate the transformation between images (homography) using RANSAC (cv.findHomography)
5. Warp one image to align it with the other (cv.warpPerspective) and visualize the result
"""
def task3(img: MatLike):
    h, w = img.shape[:2]
    M = cv.getRotationMatrix2D((w / 2, h / 2), 15, 1.1)  # rotate + scale
    img2 = cv.warpAffine(img, M, (w, h))
    gray1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    det = cv.SIFT.create()
    norm = cv.NORM_L2

    k1, d1 = det.detectAndCompute(gray1, None)
    k2, d2 = det.detectAndCompute(gray2, None)
    # Alternate between these for observation
    # bf = cv.BFMatcher(norm) #
    bf = cv.FlannBasedMatcher()
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    match_vis = cv.drawMatches(img, k1, img2, k2, good[:60], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if len(good) >= 4:
        src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)

        if H is not None:
            warped = cv.warpPerspective(img, H, (w, h))
            overlay = cv.addWeighted(img2, 0.5, warped, 0.5, 0)


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(match_vis, cv.COLOR_BGR2RGB))
    plt.title("Matches")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.axis("off")

    plt.show()


"""
# Line Detection using the Hough Tranform
1. Use Canny edge detector to extract edges from an image with clear linear features (e.g. the flowers image)
2. Detect lines using the probabilistic Hough Transform (cv.HoughLinesP)
3. Experiment with parameters: threshold, minimum line length, and maximum line gap 
"""
def task4(img: MatLike):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv.Canny(blur, 50, 150)
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                           threshold=80, minLineLength=40, maxLineGap=10)
    out = img.copy()

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv.line(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB))
    plt.title("Hough Lines")
    plt.axis("off")

    plt.show()

'''
todo
- Try using ORB instead of SIFT for image registration and compare the performance.
- Combine the methods into a small pipeline (e.g., corner detection → registration → line detection).
- Try detecting corners and tracking them between frames in a short video.
'''


if __name__ == '__main__':
    main()
