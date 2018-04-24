from pathlib import Path
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from skimage import feature
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import svm
import math

from sklearn.neural_network import MLPClassifier

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
    labels = []
    for folder in folders:
        faces = os.listdir(dirFaces + "/" + folder)
        print(dirFaces + folder)
        for face in faces:
            img = cv2.imread(dirFaces + folder + "/" + face)
            if img is None:
                print("¯\_(ツ)_/¯ Unable to load " + dirFaces + folder + "/" + face)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape
            w = size[0]
            h = size[1]
            grayVector = gray.reshape(w * h)
            try:
                testMatrix = np.vstack((testMatrix, grayVector))
                labels.append(folder + "/" + face)
            except:
                testMatrix = grayVector
                print(len(grayVector))
                labels.append(folder + "/" + face)
    print("Computing mean and Eigen vectors")
    mean, eigenVectors = cv2.PCACompute(testMatrix, mean=None, maxComponents=len(testMatrix))

    # averageFace = mean.reshape(size)
    # cv2.imwrite(dirFaces+"/average.jpg", averageFace)
    print("Computing weights")
    all = cv2.PCAProject(testMatrix, mean, eigenVectors)
    picklePCA(all, labels)
    return all


def picklePCA(pca, labels):
    all_pickle = {}
    print("Pickling")
    for i in range(0, int(len(labels) / 5)):
        all_pickle[labels[i]] = pca[i]

    pickle.dump(all_pickle, open('../cache/PCA.pickle', 'wb'))


def doTheHOG(dirFaces):
    hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
    hog_histograms = []
    labels = []
    folders = os.listdir(dirFaces)
    for folder in folders:
        faces = os.listdir(dirFaces + "/" + folder)
        print(dirFaces + folder)
        for face in faces:
            img = cv2.imread(dirFaces + folder + "/" + face, 0)
            if img is None:
                print("¯\_(ツ)_/¯ Unable to load " + dirFaces + folder + "/" + face)
                continue
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            labels.append(folder + "/" + face)
            histogram = hog.compute(img)
            hog_histograms.append(np.asarray(histogram))
    hog = []
    for hist in hog_histograms:
        tmp = []
        for i in range(0, len(hist)):
            print(hist[i][0])
            tmp.append(hist[i][0])
        hog.append(tmp)
    pickleHOG(hog, labels)
    return hog


def pickleHOG(hog, labels):
    all_pickle = {}
    print("Pickling")
    for i in range(0, int(len(labels) / 5)):
        all_pickle[labels[i]] = hog[i]

    pickle.dump(all_pickle, open('../cache/HOG.pickle', 'wb'))


def doTheLPB(dirFaces):
    folders = os.listdir(dirFaces)
    numpoints = 24
    radius = 8
    eps = 1e-7
    lbp_histograms = []
    for folder in folders:
        faces = os.listdir(dirFaces + "/" + folder)
        print(dirFaces + folder)
        for face in faces:
            img = cv2.imread(dirFaces + folder + "/" + face)
            if img is None:
                print("¯\_(ツ)_/¯ Unable to load " + dirFaces + folder + "/" + face)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = feature.local_binary_pattern(gray, numpoints, radius, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, numpoints + 3),
                                     range=(0, numpoints + 2))

            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            lbp_histograms.append(hist)
    return lbp_histograms


def doTheBow(inData):
    dictSize = 32
    bow = cv2.BOWKMeansTrainer(dictSize)
    print("Initializing Bag of words data")
    for data in inData:
        for i in range(0, 36):
            bow.add(data[i * 225:225 + (i * 225)])
        break
    print("Clustering BOW data")
    dictionary = bow.cluster()

    return dictionary


def calculateDistances(classifier):
    alldata = pickle.load(open('../cache/' + classifier + '.pickle', 'rb'))
    eOK = 0
    mOK = 0
    picNo = 1
    all = 0
    for pic1 in alldata:
        print(str(picNo / len(alldata) * 100))
        picNo += 1
        for pic2 in alldata:
            all += 1
            data_1 = np.asarray(alldata[pic1])
            data_2 = np.asarray(alldata[pic2])
            euclid = math.sqrt(sum([(a - b) ** 2 for a, b in zip(data_1, data_2)]))
            manhatan = np.sum(np.abs(data_1 - data_2))
            if euclid < 10:
                if pic1[:3] == pic2[:3]:
                    eOK += 1
            if manhatan < 500:
                if pic1[:3] == pic2[:3]:
                    mOK += 1

    e = (eOK / all) * 100
    m = (mOK / all) * 100

    print(classifier + ' euclid: ' + str(e) + " %")
    print(classifier + ' manhatan: ' + str(m) + " %")


def doTheMLP(clasifier, numlayer):
    alldata = pickle.load(open('../cache/' + clasifier + '.pickle', 'rb'))
    cnt = 0
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for picture in alldata:
        if cnt is 15:
            cnt = 0
        if cnt < 10:
            train_data.append(np.asarray(alldata[picture]))
            train_labels.append(picture[:3])
            cnt += 1
        elif cnt < 15:
            test_data.append(np.asarray(alldata[picture]))
            test_labels.append(picture[:3])
            cnt += 1

    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(numlayer, numlayer, numlayer), random_state=1, max_iter=1000)
    clf.fit(train_data, train_labels)
    print("training done")
    labels = clf.predict(test_data)
    test_labels = np.asarray(test_labels)
    correct = np.count_nonzero(labels == test_labels)
    # print(labels)
    # print(test_labels)
    # for label in labels:
    #     if label in test_labels:
    #         correct+=1

    print("MLP: "+str(((correct / len(labels)) * 100))+"%")

def doTheSVM(classifier):
    alldata = pickle.load(open('../cache/' + classifier + '.pickle', 'rb'))
    cnt = 0
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for picture in alldata:
        if cnt is 15:
            cnt = 0
        if cnt < 10:
            train_data.append(np.asarray(alldata[picture]))
            train_labels.append(picture[:3])
            cnt += 1
        elif cnt < 15:
            test_data.append(np.asarray(alldata[picture]))
            test_labels.append(picture[:3])
            cnt += 1
    clf = svm.SVC(kernel='rbf')
    clf.fit(train_data, train_labels)
    print("training done")
    labels = clf.predict(test_data)
    test_labels = np.asarray(test_labels)
    correct = np.count_nonzero(labels == test_labels)
    print("SVM: " + str(((correct / len(labels)) * 100)) + "%")

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
    # pca = doTheHOG(outDir)
    # pca = doThePCA(outDir)
    # calculateDistances("HOG")
    # calculateDistances("PCA")
    doTheMLP("HOG", 12)
    doTheSVM("HOG")


if __name__ == "__main__":
    main()
