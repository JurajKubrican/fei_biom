from pathlib import Path

import cv2
import numpy as np
import pickle as p

from os import listdir

fileDir = '../cache/extract/ear.zip/ucho/'
dumpFile = '../cache/marked/ear.zip/rects.pickle'
outDir = '../cache/ellipse/ear.zip/'
classifier = '../cascades/haarcascade_mcs_leftear.xml'

margin = 25

Path(outDir).mkdir(parents=True, exist_ok=True)


def preprocess(file):
    img = cv2.imread(fileDir + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 10)
    return gray, img


def diagonal(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_contours(processed, orig, rect, filename):
    # print(gray.shape)

    ret, canny = cv2.threshold(processed, 160, 250, 0)
    canny = cv2.Canny(canny, 56, 90, apertureSize=3, L2gradient=False)
    # print(gray.shape)

    _, cnts, hierarchy = cv2.findContours(image=canny,
                                          mode=cv2.RETR_CCOMP,
                                          method=cv2.CHAIN_APPROX_NONE)

    cnts = np.array(cnts)
    minLen = 1000
    best = 0
    allEllipses = orig.copy()
    len2 = diagonal((0, 0), (rect[2], rect[3]))
    for i, cnt in enumerate(cnts):

        if (cnt.shape[0] < 7):
            continue

        (c1, c2, angle) = cv2.fitEllipse(cnt)

        len = diagonal((0, 0), c2)
        # print(len, len2)

        if np.sqrt(len ** 2 + len2 ** 2) < minLen:
            best = i

        allEllipses = cv2.ellipse(allEllipses,
                                  (c1, c2, angle),
                                  color=(0, 255, 0),
                                  thickness=3)
        # cv2.imshow('all', allEllipses)
        # cv2.waitKey()

    # cv2.waitKey()

    (c1, c2, angle) = cv2.fitEllipse(cnts[best])
    # print(filename + ' - ')
    # print(c1, c2, diagonal(c1, c2))

    orig = cv2.ellipse(orig,
                       (c1, c2, angle),
                       color=(0, 255, 0),
                       thickness=3)

    cv2.imwrite(outDir + file, orig)


def detect(file):
    processed, orig = preprocess(file)
    dump = p.load(open(dumpFile, 'rb'))
    find_contours(processed, orig, dump[file], file)


files = listdir(fileDir)

detected = []
for file in files:
    if (file == "explain2.txt"):
        continue

    im = detect(file)
    # break

cv2.waitKey()
