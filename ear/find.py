from pathlib import Path

import cv2

from os import listdir

fileDir = '../cache/extract/ear.zip/ucho/'
outDir = '../cache/marked/ear.zip/'
classifier = '../cascades/haarcascade_mcs_leftear.xml'

margin = 5

Path(outDir).mkdir(parents=True, exist_ok=True)


def detect(file):
    left_ear_cascade = cv2.CascadeClassifier(classifier)

    if left_ear_cascade.empty():
        raise IOError('classifier xml not found :/')

    img = cv2.imread(fileDir + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRY)
    left_ear = left_ear_cascade.detectMultiScale(gray, 1.15, 5,
                                                 # minSize=(130, 200)
                                                 )

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
