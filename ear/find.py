from pathlib import Path

import cv2

from os import listdir

fileDir = '../cache/extract/ear.zip/ucho/'
outDir = '../cache/marked/ear.zip/'
classifier = '../cascades/haarcascade_mcs_leftear.xml'

margin = 15

Path(outDir).mkdir(parents=True, exist_ok=True)


def detect(file):
    left_ear_cascade = cv2.CascadeClassifier(classifier)

    if left_ear_cascade.empty():
        raise IOError('classifier xml not found :/')

    img = cv2.imread(fileDir + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    left_ear = left_ear_cascade.detectMultiScale(gray, 1.15, 5)

    if not len(left_ear):
        for scale in range(6, 14):
            gray = cv2.resize(gray, (0, 0), fx=(scale / 10), fy=1)
            left_ear = left_ear_cascade.detectMultiScale(gray, 1.15, 5)
            if len(left_ear):
                break

    if not len(left_ear):
        for deg in range(0, -45):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rows, cols = gray.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
            gray = cv2.warpAffine(gray, M, (cols, rows))
            left_ear = left_ear_cascade.detectMultiScale(gray, 1.15, 5)
            if len(left_ear):
                break

    if not len(left_ear):
        for deg in range(0, 45):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rows, cols = gray.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
            gray = cv2.warpAffine(gray, M, (cols, rows))
            left_ear = left_ear_cascade.detectMultiScale(gray, 1.15, 5)
            if len(left_ear):
                break


    if not len(left_ear):
        cv2.imwrite(outDir + file, img)
        return ()

    bestX = 0
    bestY = 0
    maxW = 0
    maxH = 0
    for (x, y, w, h) in left_ear:
        if h * w > maxH * maxW:
            bestX = x
            bestY = y
            maxW = w
            maxH = h

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(img, (bestX - margin, bestY - margin), (bestX + maxW + 2 * margin, bestY + maxH + 2 * margin),
                  (0, 255, 0), 3)

    cv2.imwrite(outDir + file, img)
    return left_ear


files = listdir(fileDir)

detected = []
for file in files:
    if (file == "explain2.txt"):
        continue

    im = detect(file)
    if im.__len__():
        detected.append(im)

print('uspesnost: ' + str(detected.__len__() / files.__len__()))
print('in: ' + str(files.__len__()))
print('out: ' + str(detected.__len__()))

exit(0)
