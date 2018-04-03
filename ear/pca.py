from pathlib import Path

import cv2
import numpy as np
from os import listdir

fileDir = '../cache/ellipse/ear.zip/'
outDir = '../cache/lbp/'
dumpFile = outDir + 'ear.pickle'

Path(outDir).mkdir(parents=True, exist_ok=True)


def preprocess(file):
    img = cv2.imread(file,0)
    return img


files = listdir(fileDir)

detected = []
lbp_source = []
mean = np.zeros(200 * 150 * 128)
size = (200, 150)
pca_source = []
eigenvectors = []
for file in files:
    if (file == "explain2.txt"):
        continue

    im = preprocess(fileDir + file)
    shape = im.shape
    pca_source.append(im.flatten())

# pca_source = np.transpose(pca_source)

print('counting pca')
mean, eigvec = cv2.PCACompute(np.asarray(pca_source), mean=None)
print('done')

average = mean.reshape(size)

first = eigvec[0].reshape(size)
cv2.imshow("/average.jpg", average + first)
cv2.waitKey()
