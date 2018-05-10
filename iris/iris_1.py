import cv2
import numpy as np

import os
import os.path

import math
# img = cv2.imread('../cache/extract/iris.zip/009/1/009_1_1.bmp',0)

# img = cv2.medianBlur(img,5)
# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
# cv2.equalizeHist(img,cimg);
from IPython.core.display import Math


def c_detect(img, cimg):

    imgA = cv2.medianBlur(img, 5)

    circles = cv2.HoughCircles(imgA, cv2.HOUGH_GRADIENT, 1, 500,
                               param1=200, param2=1, minRadius=25, maxRadius=0)

    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        print("ac ")
        print(i)
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        r = i[2]
        print(r)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 2)
        # cv2.circle(cimg,(100,100),20,(0,0,255),2)
        stred = [i[0], i[1]]


    imgB = cv2.medianBlur(img, 5)
    imgB  = cv2.equalizeHist(imgB)



    circles = cv2.HoughCircles(imgB, cv2.HOUGH_GRADIENT, 3, 600,
                               param1=100, param2=32, minRadius=r + 45, maxR6adius= math.floor(r* 3))
    circles = np.uint16(np.around(circles))



    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        print(i[0] + ' - ' + i[1])

    #////////



def detect_outer(img, cimg) :


    imgB = cv2.medianBlur(img, 5)
    imgB = cv2.equalizeHist(imgB)

    circles = cv2.HoughCircles(imgB, cv2.HOUGH_GRADIENT, 2, 10,
                               param1=200, param2=1, minRadius=200, maxRadius=400)
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 255), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        print(i[0] + ' - ' + i[1])







def c_detect_in_file(path):
    file_dir = os.listdir(path)
    for file in file_dir:
        sub_path = os.path.join(path, file)
        files = os.listdir(sub_path)
        for img_f in files:
            if (img_f == "Thumbs.db"): continue
            img_path = os.path.join(sub_path, img_f)
            print(img_f)
            print(img_path)

            img = cv2.imread(img_path, 0)

            # img = cv2.medianBlur(img, 5)
            fimg = cv2.equalizeHist(img)
            cimg = cv2.cvtColor(fimg, cv2.COLOR_GRAY2BGR, 0)

            try:
             # c_detect(img, cimg)
                detect_outer(img, cimg)
            except Exception as e:
                print('Subor sa nam zlozil > ', img_f, ':', e)



            cv2.imshow('detected circles', cimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()





path = "C:/Users/Erik/PycharmProjects/fei_biom/cache/extract/iris.zip"
file_dirs = os.listdir(path)

for dir_path in file_dirs:
    sub_path = os.path.join(path, dir_path)
    c_detect_in_file(sub_path)
