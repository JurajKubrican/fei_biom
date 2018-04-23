from pathlib import Path

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from matplotlib import pyplot as plt
from os import listdir
import pickle

file_dir: str = '../cache/ellipse/ear.zip/'
out_dir: str = '../cache/'
dump_file: str = out_dir + 'ear-lbp.pickle'

Path(out_dir).mkdir(parents=True, exist_ok=True)


def lbpify(file):
    # processed, orig = preprocess(file)

    img_src = cv2.imread(file, 0)
    shapesX = [
        (0, 50),
        (50, 100),
        (10, 150),
        (150, 200),
    ]
    shapesY = [
        (0, 50),
        (50, 100),
        (10, 150),
    ]
    numpoints = 9
    radius = 3
    eps = 1e-7
    plti = 0

    hists = []
    for i in range(4):
        for j in range(3):
            img = img_src[shapesX[i][0]:shapesX[i][1], shapesY[j][0]:shapesY[j][1]]
            transformed_img = local_binary_pattern(img, numpoints,
                                                   radius, method="uniform")
            (hist, _) = np.histogram(transformed_img.ravel(),
                                     bins=np.arange(0, 0 + 3),
                                     range=(0, 0 + 2))

            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            hists.append(hist)
    #         plti += 1
    #         plt.subplot(4, 3, plti)
    #         plt.hist(transformed_img.flatten(), 256, [0, 10], color='r')
    # plt.show()
    return hists


files = listdir(file_dir)

detected = []
lbp_source = []
mean = np.zeros(200 * 150 * 3 * 128)
size = (200, 150, 3)
hists = {}

for file in files:
    if (file == "explain2.txt"):
        continue

    print(file)
    hists[file] = lbpify(file_dir + file)

pickle.dump(hists, open(dump_file, 'wb'))
