from pathlib import Path

import cv2
import numpy as np
import os

fileDir = '../cache/extract/face.zip/gt_db/'
outDir = '../cache/marked/face.zip/gt_db/'

Path(outDir).mkdir(parents=True, exist_ok=True)


def detectFace(file):
    faceCascade = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_default.xml')
    eyesCascade = cv2.CascadeClassifier('../cascades/haarcascade_eye.xml')

    if faceCascade.empty():
        raise IOError('Unable to load the face cascade classifier xml file ¯\_(ツ)_/¯')
    if eyesCascade.empty():
        raise IOError('Unable to load the eyes cascade classifier xml file ¯\_(ツ)_/¯')
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detectedFaces = faceCascade.detectMultiScale(gray, 1.3, 5)
    if (len(detectedFaces) != 1):
        return
    for (x, y, w, h) in detectedFaces:
        # cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        detectedEyes = eyesCascade.detectMultiScale(face, 1.3, 5)
        if (len(detectedEyes) != 2):
            return
        for (ex, ey, ew, eh) in detectedEyes:
            cv2.circle(face, (ex + int(eh / 2), ey + int(eh / 2)), int(eh / 2), (255, 0, 0), 2)

        if (detectedEyes[0][0] < detectedEyes[1][0]):
            leftEye = detectedEyes[0]
            rightEye = detectedEyes[1]
        else:
            rightEye = detectedEyes[0]
            leftEye = detectedEyes[1]

    aligned = alignFace(gray, leftEye, rightEye)
    detectedFace = faceCascade.detectMultiScale(aligned, 1.3, 5)
    for (x, y, w, h) in detectedFace:
        cv2.rectangle(aligned, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx = x - 20
        cy = y - 20
        face = aligned[cy:cy + h + 40, cx:cx + w + 40]

    return face


def alignFace(face, leftEye, rightEye):
    leftEyeCenter = (leftEye[0] + int(leftEye[2] / 2), leftEye[1] + int(leftEye[2] / 2))
    rightEyeCenter = (rightEye[0] + int(rightEye[2] / 2), rightEye[1] + int(rightEye[2] / 2))

    dY = leftEyeCenter[1] - rightEyeCenter[1]
    dX = leftEyeCenter[0] - rightEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    center = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
              (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    (faceW, faceH) = face.shape

    M = cv2.getRotationMatrix2D(center, angle, 1)
    output = cv2.warpAffine(face, M, (faceW, faceH))
    return output

def doThePCA(dirFaces):
    testMatrix = None
    folders = os.listdir(dirFaces)
    for folder in folders:
        faces = os.listdir(dirFaces+"/"+folder)
        print(dirFaces+folder)
        for face in faces:
            img = cv2.imread(dirFaces+folder+"/"+face)
            if img is None:
                print("¯\_(ツ)_/¯ Unable to load "+ dirFaces+folder+"/"+face)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape
            w=size[0]
            h=size[1]
            grayVector = gray.reshape(w*h)
            try:
                testMatrix = np.vstack((testMatrix, grayVector))
            except:
                testMatrix = grayVector
    print("Computing mean and Eigen vectors")
    mean, eigenVectors = cv2.PCACompute(testMatrix, mean=None, maxComponents=len(testMatrix))

    # averageFace = mean.reshape(size)
    # cv2.imwrite(dirFaces+"/average.jpg", averageFace)
    print("Computing weights")
    all = cv2.PCAProject(testMatrix, mean, eigenVectors)
    return all

def doTheHOG(dirFaces):
    hog = cv2.HOGDescriptor()
    hog_histograms = []
    folders = os.listdir(dirFaces)
    for folder in folders:
        faces = os.listdir(dirFaces + "/" + folder)
        print(dirFaces + folder)
        for face in faces:
            img = cv2.imread(dirFaces + folder + "/" + face)
            if img is None:
                print("¯\_(ツ)_/¯ Unable to load " + dirFaces + folder + "/" + face)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            histogram = hog.compute(gray)
            hog_histograms.append(histogram)
    return hog_histograms



def main():
    # test = detectFace('../cache/extract/face.zip/gt_db/s01/02.jpg')
    # folders = os.listdir(fileDir)
    # totalPictures = 0
    # facesDetected = 0
    # for folder in folders:
    #     pictures = os.listdir(fileDir + folder)
    #     if not (os.path.exists(outDir + folder)):
    #         os.mkdir(outDir + folder)
    #     print(fileDir + folder)
    #     for picture in pictures:
    #         face = detectFace(fileDir + folder + '/' + picture)
    #         totalPictures += 1
    #         if not (face is None):
    #             cv2.imwrite(outDir + '/' + folder + '/' + picture, face)
    #             facesDetected += 1
    #
    # percent = (facesDetected*100)/totalPictures
    # print("Detected "+str(percent)+"% of faces")
    # doThePCA(outDir)
    doTheHOG(outDir)


if __name__ == "__main__":
    main()
