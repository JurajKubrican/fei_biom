from pathlib import Path

import cv2

from os import listdir

fileDir = '../cache/extract/ear.zip/ucho/'
outDir = '../cache/marked/face.zip/gt_db/'
classifier = '../cascades/haarcascade_mcs_leftear.xml'

Path(outDir).mkdir(parents=True, exist_ok=True)

def detectFace(file):
    face = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_default.xml')
    eyes = cv2.CascadeClassifier('../cascades/haarcascade_eye.xml')

    if face.empty():
        raise IOError('Unable to load the face cascade classifier xml file ¯\_(ツ)_/¯')
    if eyes.empty():
        raise IOError('Unable to load the eyes cascade classifier xml file ¯\_(ツ)_/¯')
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detedtedFaces = face.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in detedtedFaces:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        detectedEyes = eyes.detectMultiScale(face, 1.3, 5)
        for (ex, ey, ew, eh) in detectedEyes:
            cv2.circle(face, (ex + int(eh / 2), ey + int(eh / 2)), int(eh / 2), (255, 0, 0), 2)

    cv2.imshow('face', gray)
    cv2.waitKey()
    cv2.destroyAllWindows()

def main():
    detectFace('../cache/extract/face.zip/gt_db/s01/09.jpg')

if __name__ == "__main__":
    main()