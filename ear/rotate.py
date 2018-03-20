from pathlib import Path

import cv2
import numpy as np

from os import listdir

fileDir = '../cache/extract/ear.zip/ucho/'
outDir = '../cache/marked/ear.zip/'
classifier = '../cascades/haarcascade_mcs_leftear.xml'

margin = 25

Path(outDir).mkdir(parents=True, exist_ok=True)


def find_contours(gray):
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.Canny(gray, 50, 80, apertureSize=3, L2gradient=False)
    # print(gray.shape)
    # # ret, gray = cv2.threshold(gray, 160, 250, 0)
    # print(gray.shape)

    img, cnt, hierarchy = cv2.findContours(image=gray,
                                           mode=cv2.RETR_TREE,
                                           method=cv2.CHAIN_APPROX_NONE)

    cnt = np.array(cnt[0])
    hierarchy = np.array(hierarchy)
    # print(gray.shape)
    # print(img.shape)
    # print(cnt.shape)
    # print(cnt)
    # print(hierarchy)
    # .shape)
    # cv2.imshow('canny', img)
    # cv2.waitKey()
    gray = cv2.drawContours(gray, cnt, contourIdx=-1, color=(255, 0, 0))
    cv2.imshow('gray', gray)
    cv2.imshow('img', img)

    ellipse = cv2.fitEllipse(cnt)
    print(ellipse)
    ellipse_img = cv2.ellipse(img,
                              center=ellipse[0],
                              axes=ellipse[1],
                              angle=ellipse[2],
                              color=(0, 255, 0))
    # Draw both contours onto the separate images

    # cnt = contours[0]
    # cnt = np.array(cnt, dtype=np.uint32)

    cv2.imshow('sss', ellipse_img)
    cv2.waitKey()


def detect(file):
    left_ear_cascade = cv2.CascadeClassifier(classifier)

    if left_ear_cascade.empty():
        raise IOError('classifier xml not found :/')

    img = cv2.imread(fileDir + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # gray = cv2.blur(gray, 1)

    cv2.imshow("Ucho", gray)

    find_contours(img)


files = listdir(fileDir)

detected = []
im = detect(files[10])
for file in files:
    if (file == "explain2.txt"):
        continue

    im = detect(file)
    break

cv2.waitKey()
