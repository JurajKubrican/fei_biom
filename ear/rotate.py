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


# def generate_gauss():
#     x, y = np.meshgrid(np.linspace(-1, 1, 300), np.linspace(-1, 1, 400))
#     d = np.sqrt(x * x + y * y)
#     sigma, mu = 0.40, 0.50
#     g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
#     mask = np.ones((400, 300), 'float64')
#     # cv2.imshow('dd',g)
#     # cv2.waitKey()
#     mask[200:400, 230:300] = g[200:400, 230:300]
#     mask[330:400, 100:300] = g[330:400, 100:300]
#     # mask[200:400, 150:300] = g[200:400, 150:300]
#     return mask


def ellipse_mask(image, ellipse):
    (center, size, angle) = ellipse

    mask = np.zeros_like(image)
    rows, cols, _ = mask.shape
    # create a white filled ellipse
    cv2.ellipse(mask, ellipse,
                color=(255, 255, 255), thickness=-1)
    image = np.bitwise_and(image, mask)
    # center
    tr = np.float32([
        [1, 0, 150 - center[0]],
        [0, 1, 200 - center[1]],
    ])
    image = cv2.warpAffine(image, tr, (300, 400))

    # rotate
    rot = cv2.getRotationMatrix2D((150, 200), angle, 1)
    image = cv2.warpAffine(image, rot, (300, 400))

    # stretch
    image = cv2.resize(image, (0, 0), fx=(150 / size[0]), fy=(200 / size[1]))

    c = (int(image.shape[0] / 2), int(image.shape[1] / 2))

    image = image[c[0] - 100: c[0] + 100, c[1] - 75: c[1] + 75]

    return image


def preprocess(file):
    img = cv2.imread(fileDir + file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = np.asarray(gray *  generate_gauss(),'uint8')
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7, 3), 15)
    return gray, img


def distance(p1, p2):
    return np.sqrt((p1[0] - p1[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_contours(processed, orig, rect):
    ret, canny = cv2.threshold(processed, 160, 250, 0)
    canny = cv2.Canny(canny, 56, 90, apertureSize=3, L2gradient=False)
    _, cnts, _ = cv2.findContours(image=canny,
                                  mode=cv2.RETR_CCOMP,
                                  method=cv2.CHAIN_APPROX_NONE)

    cnts = np.array(cnts)
    minDist = 10000000000
    maxArea = 0
    best = -1
    all_ellipses = orig.copy()
    filtered_ellipses = orig.copy()
    best_ellipse = ((50, 50), (50, 50), 0)

    print(file)
    for i, cnt in enumerate(cnts):

        if (cnt.shape[0] < 7):
            continue

        (center, size, angle) = cv2.fitEllipse(cnt)

        areaRect = (rect[2] * rect[3])
        areaEllipse = (size[0] * size[1])

        cv2.ellipse(all_ellipses,
                    (center, size, angle),
                    color=(0, 0, 255),
                    thickness=2)

        if areaEllipse < 15000:
            continue

        if center[0] < rect[0] \
                or center[1] < rect[1] \
                or center[0] > 300 \
                or center[1] > 400:
            continue

        cv2.ellipse(filtered_ellipses,
                    (center, size, angle),
                    color=(0, 255, 0),
                    thickness=3)

        bulhar_dist = np.sqrt(abs(areaRect - areaEllipse)) + distance(center, (rect[0], rect[1]))
        bulhar_dist = max(abs(areaRect - areaEllipse), distance(center, (rect[0], rect[1])))

        if bulhar_dist < minDist:
            best_ellipse = (center, size, angle)
            minDist = bulhar_dist
        # if (maxArea < areaEllipse):
        #     maxArea = areaEllipse
        #     best = i

    # cv2.imshow('all', allEllipses)
    # cv2.waitKey()

    image = ellipse_mask(orig, best_ellipse)

    # c aitKey()
    #

    # #
    #

    # rot = cv2.getRotationMatrix2D((200,150), angle, 1)
    # print(rot)
    #

    cv2.imwrite(outDir + file, image)
    # cv2.imwrite(outDir + file + ".bmp", filtered_ellipses)
    #
    # cv2.imwrite(outDir + file, all_ellipses)
    # cv2.imwrite(outDir + file + ".bmp", filtered_ellipses)


def detect(file):
    processed, orig = preprocess(file)
    dump = p.load(open(dumpFile, 'rb'))

    find_contours(processed, orig, dump[file])
    # find_contours(processed, processed, dump[file], file)


files = listdir(fileDir)

detected = []
for file in files:
    if (file == "explain2.txt"):
        continue

    # im = detect("01-1.bmp")
    # break
    print(file)
    im = detect(file)

# cv2.waitKey()
