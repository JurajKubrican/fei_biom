from pathlib import Path

import cv2
import numpy as np

from os import listdir

fileDir = '../cache/extract/ear.zip/ucho/'
outDir = '../cache/marked/ear.zip/'
classifier = '../cascades/haarcascade_mcs_leftear.xml'

margin = 25

Path(outDir).mkdir(parents=True, exist_ok=True)


def find_contours(img):
    canny = cv2.Canny(img, 50, 80)

    contours = cv2.findContours(image=canny,
                                mode=cv2.RETR_TREE,
                                method=cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]

    print(cnt)

    ellipse = cv2.fitEllipse(cnt)
    ellipse_img = cv2.ellipse(img, ellipse, (0, 255, 0), 2)

    cv2.imshow('sss', ellipse_img)
    cv2.waitKey()


def detect(file):
    left_ear_cascade = cv2.CascadeClassifier(classifier)

    if left_ear_cascade.empty():
        raise IOError('classifier xml not found :/')

    img = cv2.imread(fileDir + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cv2.imshow("Ucho", gray)

    find_contours(img)


files = listdir(fileDir)

detected = []
for file in files:
    if (file == "explain2.txt"):
        continue

    im = detect(file)
    break

cv2.waitKey()
