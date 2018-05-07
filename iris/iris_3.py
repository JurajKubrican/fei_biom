import cv2

import os
import os.path
import decimal
import math
from matplotlib import pyplot as plt
import numpy as np

import numpy as np
import cv2
import pickle



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def load_images(path):
    images = []
    for root,dirs,filename in os.walk(path):
        for image in filename:
            if os.path.splitext(image)[1] == ".bmp":
                images.append(os.path.join(root, image))
    return images


def load_coordinates(path):
    coords_array = []


    with open(path) as f:
        f.readline()
        lines = f.readlines();
        print(lines)
        for line in lines:
            coord = []
            data = line.strip().split(",")
            for value in data:
                coord.append(value)
            coords_array.append(coord)

    return coords_array


import decimal

def drange(x, y, jump):
    i = x
    while i < y:
        yield i
        i += jump

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step



# ksize – Size of the filter returned.
# sigma – Standard deviation of the gaussian envelope.
# theta – Orientation of the normal to the parallel stripes of a Gabor function.
# lambd – Wavelength of the sinusoidal factor.
# gamma – Spatial aspect ratio.
# psi – Phase offset.
# ktype – Type of filter coefficients. It can be CV_32F or CV_64F .


def build_kern(kSize):
        kern = cv2.getGaborKernel((kSize, kSize), 1.0, 0.5, 12.0, 0.5, 1.7, ktype=cv2.CV_32F)                #sigma - Gaussian standard deviation.
        kern /= 1.5 * kern.sum()

        return kern


def build_kern2(kSize):
        kern = cv2.getGaborKernel((kSize, kSize), 1.0, 0.5  , 12.0, 0.5, -1.7, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()

        return kern


# src – input image.
# dst – output image of the same size and the same number of channels as src.
# ddepth –
# desired depth of the destination image; if it is negative, it will be the same as src.depth(); the following combinations of src.depth() and ddepth are supported:
# src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
# src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
# src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
# src.depth() = CV_64F, ddepth = -1/CV_64F
# when ddepth=-1, the output image will have the same depth as the source.
#
# kernel – convolution kernel (or rather a correlation kernel), a single-channel floating point matrix; if you want to apply different kernels to different channels, split the image into separate color planes using split() and to2D them individually.
# anchor – anchor of the kernel that indicates the relative position of a filtered point within the kernel; the anchor should lie within the kernel; default value (-1,-1) means that the anchor is at the kernel center.
# delta – optional value added to the filtered pixels before storing them in dst.
# borderType – pixel extrapolation method (see borderInterpolate for details).


def to2D(img, kernel):
        d_value = np.zeros_like(img)
        processed_img = cv2.filter2D(img, cv2.CV_8UC1, kernel)                       #Convolves an image with the kernel.
        np.maximum(d_value, processed_img, d_value)                                      #porovnanie matic a vybranie najvyssich hodnot
        return d_value



def gaboralization(path):

    for path_img in path:

        img = cv2.imread(path_img, 0)
        # print(path_img.split('\\')[3])
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        #
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
        #
        # cv2.imshow('image', img)
        # cv2.imshow('filtered image', filtered_img)
        #
        # h, w = g_kernel.shape[:2]
        # g_kernel = cv2.resize(g_kernel, (3 * w, 3 * h), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('gabor kernel (resized)', g_kernel)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()




        # cv2.imshow("Mask", img)
        # cv2.waitKey()

        # cv2.imshow("Input image", img)
        # cv2.waitKey()

        img = cv2.equalizeHist(img)
        kern_s = 17
        kernel = build_kern(kern_s)
        kernel2 = build_kern2(kern_s)

        img = to2D(img, kernel)
        unwrapped2 = to2D(img, kernel2)
        unwrappedDiff = img - unwrapped2


        _, img = cv2.threshold(unwrappedDiff, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # cv2.imshow("Binarized image", img)
        # cv2.waitKey()

        filename_out = "C:/Users/Erik/PycharmProjects/fei_biom/iris/binarize/" + path_img.split('\\')[3]
        # print(filename_out)
        cv2.imwrite(filename_out, img)







path = "C:/Users/Erik/PycharmProjects/fei_biom/iris/polar"
surr_path = "./iris_bounding_circles.csv"

images = load_images(path)
coordinates = load_coordinates(surr_path)



for file in path:
    gaboralization(images)


