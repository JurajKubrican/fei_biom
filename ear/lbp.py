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


def lbpify(img_src):
    # processed, orig = preprocess(file)

    shapesX = [
        (0, 100),
        (100, 200),
        (200, 300),
        (300, 400),
    ]
    shapesY = [
        (0, 50),
        (50, 100),
        (100, 150),
        (150, 200),
        (200, 250),
    ]
    numpoints = 9
    radius = 3
    eps = 1e-7

    hists = []
    for i in range(4):
        for j in range(3):
            img = img_src[shapesX[i][0]:shapesX[i][1], shapesY[j][0]:shapesY[j][1]]
            if img.shape != (50, 100):
                continue
            transformed_img = local_binary_pattern(img, numpoints,
                                                   radius, method="uniform")
            (hist, _) = np.histogram(transformed_img.ravel(),
                                     bins=np.arange(0, 0 + 3),
                                     range=(0, 0 + 2))

            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            hists.append(hist)

    return hists


files = listdir(file_dir)
temp = dict()
for file in files:
    if file == "explain2.txt":
        continue

    label = file[:2]
    img = cv2.imread(file_dir + file, 0)
    img = cv2.equalizeHist(img, 0)
    if img.shape == (200, 150) or img.shape == (400, 250):
        data = np.asarray(lbpify(img)).flatten()
        temp.setdefault(label, []).append(data)

output = dict()
for label in temp:
    if len(temp[label]) == 4:
        output[label] = temp[label]

pickle.dump(output, open(dump_file, 'wb'))
