import pickle
from pathlib import Path

import cv2
import numpy as np
from os import listdir

file_dir = '../cache/ellipse/ear.zip/'
out_dir = '../cache/'
dump_file = out_dir + 'ear-pca.pickle'

Path(out_dir).mkdir(parents=True, exist_ok=True)

files = listdir(file_dir)

size = (200, 150)
pca_source = []
labels = []
for file in files:
    if (file == "explain2.txt"):
        continue
    labels.append(file)
    im = cv2.imread(file_dir + file, 0)
    shape = im.shape
    pca_source.append(im.flatten())

pca_source = np.asarray(pca_source)
mean, eigvec = cv2.PCACompute(pca_source, mean=None)
mean = np.asarray(mean)
average = np.asarray(mean.reshape(size), 'uint8')
eigvec = np.asarray(eigvec)
vec = cv2.PCAProject(pca_source, mean, eigvec)

output = {}

for i in range(len(labels)):
    output[labels[i]] = vec[i]

pickle.dump(output, open(dump_file, 'wb'))
