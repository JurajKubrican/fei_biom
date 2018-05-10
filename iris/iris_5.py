from __future__ import print_function

import os.path
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2


def load_images(path):
    file_dir = os.listdir(path)
    images = []
    for image in (file_dir):
        # print(image.split('_')[0] == 2)
        if os.path.splitext(image)[1] == ".bmp":
            if (image.split('_')[1] == '1'):
                pass
            else:
                images.append(os.path.join(path, image))
    print(len(images))
    return images


def print_img(images, imgx, imgy):
    simg = cv2.imread(imgx, 0)
    timg = cv2.imread(imgy, 0)
    cv2.imshow(imgx.split('/')[6], simg)
    cv2.imshow(imgy.split('/')[6], timg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def getHaming(images, masks, idx1, idx2):
    gabor1 = cv2.imread(images[idx1], 0)
    mask1 = cv2.imread(masks[idx1], 0)

    gabor2 = cv2.imread(images[idx2], 0)
    mask2 = cv2.imread(masks[idx2], 0)

    a = gabor1 * mask1 * mask2
    b = gabor2 * mask1 * mask2
    val = []
    for i in range(-10, 10):
        b2 = np.roll(b, i, axis=1)
        mat = a != b2
        score = np.average(mat)
        val.append(score)

    return min(val)


def get_mean_var(images, masks):
    persons = set()
    for img in images:
        persons.add(img.split('\\')[1].split('_')[0])
    persons = sorted(persons)
    data = []
    for x in range(0, len(images)):
        gabor1 = cv2.imread(images[x], 0)
        mask1 = cv2.imread(masks[x], 0)
        for y in range(0, len(images)):
            if x == y:
                continue
            gabor2 = cv2.imread(images[y], 0)
            mask2 = cv2.imread(images[y], 0)
            a = gabor1 * mask1 * mask2
            b = gabor2 * mask1 * mask2
            val = []
            for i in range(-10, 10):
                b2 = np.roll(b, i, axis=1)
                mat = a != b2
                score = np.average(mat)
                val.append(score)
            data.append(min(val))
    # same = []
    # different = []
    # print("vsetky", len(persons)*len(data))

    # for person in persons:
    #     # print(person)
    #     for item in data:
    #         p = item['person'].split('_')[0]
    #         c = item['compare'].split('_')[0]
    #         # print(c)
    #         if p == person and c == person:
    #             # print(item['score'])
    #             same.append(item['score'])
    #         elif p == person and c != person:
    #             different.append(item['score'])
    # print(len(different))
    # plt.title('Hamming Distance')
    # plt.hist(different, bins='auto', label='Different')
    # print('tie iste ', len(same))
    # plt.hist(same, bins=30, color='y', alpha=0.3, label='Same')
    # plt.show()

    # print(data)
    # print("priemer", np.mean(data))
    # print("rozptyl", np.var(data))
    return np.mean(data), np.var(data)
    # z = (getHamming(images,masks,ix,iy) - np.mean(data))/np.var(data)
    # return z


def test_all_iris(images, masks_data):
    mean, var = get_mean_var(images, masks)

    z_array = []
    # images_labels =

    for x in range(0, len(images)):
        for y in range(0, len(images)):
            if (x != y):
                z = (getHaming(images, masks_data, x, y) - mean) / var
                z_array.append(z)

    print("array_of_z je ", z_array)
    return z_array


def classify(ximg, yimg):
    path_binar = "C:/Users/Erik/PycharmProjects/fei_biom/iris/binarize"
    path_mask = "C:/Users/Erik/PycharmProjects/fei_biom/iris/mask_file"

    images = load_images(path_binar)
    masks = load_images(path_mask)

    # ximg = 1  # index x obrazka
    # yimg = 7  # index y obrazka
    mean, var = get_mean_var(images, masks)  # priemer a rozptyl
    z_val = (getHaming(images, masks, ximg, yimg) - mean) / var  # jedna konkretna z hodnota
    return z_val
    #
    # array_of_z = test_all_iris(images,
    #                            masks)  # vytvori pole vsetkych z hodnot s tym ze index hodnoty v poli je index obrazkov x*y++ (1*2,1*3,1*4... #rovnake indexy preskakuje
# print_img(images,ximg,yimg)
