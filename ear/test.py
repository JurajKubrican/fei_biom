#!/usr/bin/python
import cv2
import numpy as np
import sys
from numpy.linalg import eig, inv


# param is the result of canny edge detection
def process_image(img):
    # for every pixel in the image:
    for (x, y), intensity in np.ndenumerate(img):
        # if the pixel is part of an edge:
        if intensity == 255:
            # determine if the edge is similar to an ellipse
            ellipse_test(img, x, y)


def ellipse_test(img, i, j):
    # poor coding practice but what I'm doing for now
    global output, image
    i_array = []
    j_array = []
    # flood fill i,j while storing all unique i,j values in arrays
    flood_fill(img, i, j, i_array, j_array)
    i_array = np.array(i_array)
    j_array = np.array(j_array)
    if i_array.size >= 10:
        # put those values in a numpy array
        # which can have an ellipse fit around it
        array = []
        for i, elm in enumerate(i_array):
            array.append([int(j_array[i]), int(i_array[i])])
        array = np.array([array])
        print(array)
        ellp = cv2.fitEllipse(array)
        cv2.ellipse(image, ellp, (0, 0, 0))
        cv2.ellipse(output, ellp, (0, 0, 0))


def flood_fill(img, i, j, i_array, j_array):
    if img[i][j] != 255:
        return
    # store i,j values
    i_array.append(float(i))
    j_array.append(float(j))
    # mark i,j as 'visited'
    img[i][j] = 250
    # flood_fill adjacent and diagonal pixels

    (i_max, j_max) = img.shape

    if i - 1 > 0 and j - 1 > 0:
        flood_fill(img, i - 1, j - 1, i_array, j_array)
    if j - 1 > 0:
        flood_fill(img, i, j - 1, i_array, j_array)
    if i - 1 > 0:
        flood_fill(img, i - 1, j, i_array, j_array)
    if i + 1 < i_max and j + 1 < j_max:
        flood_fill(img, i + 1, j + 1, i_array, j_array)
    if j + 1 < j_max:
        flood_fill(img, i, j + 1, i_array, j_array)
    if i + 1 < i_max:
        flood_fill(img, i + 1, j, i_array, j_array)
    if i + 1 < i_max and j - 1 > 0:
        flood_fill(img, i + 1, j - 1, i_array, j_array)
    if i - 1 > 0 and j + 1 < j_max:
        flood_fill(img, i - 1, j + 1, i_array, j_array)


image = cv2.imread('../cache/extract/ear.zip/ucho/01-1.bmp', 0)
canny_result = cv2.GaussianBlur(image, (3, 3), 0)
canny_result = cv2.Canny(canny_result, 50, 80,
                         apertureSize=3, L2gradient=False)

# output is a blank images which the ellipses are drawn on
output = np.zeros(image.shape, np.uint8)
output[:] = [255]

cv2.waitKey(0)
cv2.namedWindow("Canny result:", cv2.WINDOW_NORMAL)
cv2.imshow('Canny result:', canny_result)
print("Press any key to find the edges")
cv2.waitKey(0)
print("Now finding ellipses")

process_image(canny_result)
print("Ellipses found!")
cv2.namedWindow("Original image:", cv2.WINDOW_NORMAL)
cv2.imshow('Original image:', image)

cv2.namedWindow("Output image:", cv2.WINDOW_NORMAL)
cv2.imshow("Output image:", output)
cv2.waitKey(0)
