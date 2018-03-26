from pathlib import Path

import cv2
import numpy as np
import pickle as p
from skimage import feature

from matplotlib import pyplot as plt
from os import listdir

fileDir = '../cache/extract/ear.zip/ucho/'
outDir = '../cache/lbp/'
dumpFile = outDir + 'ear.pickle'
classifier = '../cascades/haarcascade_mcs_leftear.xml'

Path(outDir).mkdir(parents=True, exist_ok=True)


def preprocess(file):
    img = cv2.imread(fileDir + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)
    return gray, img


def lbpify(file):
    # processed, orig = preprocess(file)

    img = cv2.imread(file, 0)
    numpoints = 256
    radius = 3
    eps = 1e-7

    transformed_img = feature.local_binary_pattern(img, numpoints,
                                                   radius, method="uniform")

    cv2.imshow('image', img)
    cv2.imshow('thresholded image', transformed_img)

    # hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    (hist, _) = np.histogram(transformed_img.ravel(),
                             bins=np.arange(0, 0 + 3),
                             range=(0, 0 + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)

    plt.hist(transformed_img.flatten(), 256, [0, 256], color='r')
    # plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


files = listdir(fileDir)

detected = []
for file in files:
    if (file == "explain2.txt"):
        continue

    im = lbpify(fileDir + file)

cv2.waitKey()
