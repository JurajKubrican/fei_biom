from __future__ import print_function

import cv2

import os
import os.path
import decimal
import math
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pickle
import os
import cv2
from six.moves import cPickle as pickle
from scipy.spatial import distance
from matplotlib.patches import Rectangle



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def load_images(path):
    file_dir = os.listdir(path)
    images = []
    for image in (file_dir):
        if os.path.splitext(image)[1] == ".bmp":
            images.append(os.path.join(path, image))
    print(images)
    return images

def load_coordinates(path):
    coords_array = []


    with open(path) as f:
        f.readline()
        lines = f.readlines();
        print(lines)
        for line in lines:
            coord = []
            data = line.strip().split(",")
            for value in data:
                coord.append(value)
            coords_array.append(coord)

    return coords_array


import decimal

def drange(x, y, jump):
    i = x
    while i < y:
        yield i
        i += jump

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def print_img(title, img):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.show()


# def hamming_distance(img1, img2):
#     return distance.hamming(img1, img2)

def hammingation(images, masks):
    print("wake up hammingation")
    persons = set()
    for img in images:
        persons.add(img.split('\\')[1].split('_')[0])
    persons = sorted(persons)
    print(persons)
    data = []
    print("start processing")
    poc = 0;
    val =0
    print(len(images))
    for x in range(0, len(images)):
        gabor1 = cv2.imread(images[x], 0)
        mask1 = cv2.imread(masks[x], 0)
        for y in range(0, len(images)):
            if x == y:
                continue;
            gabor2 = cv2.imread(images[y], 0)
            mask2 = cv2.imread(images[y], 0)
            # mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
            a = gabor1 * mask1 * mask2
            b = gabor2 * mask1 * mask2
            val = []
            poc = 0;
            for i in range(-10, 10):
                # print("processing "+ images[x].split('\\')[1].split('_')[0] + " with " + images[y].split('\\')[1].split('_')[0])
                b2 = np.roll(b, i, axis=1)
                mat = a != b2
                # print(mat)
                score = np.average(mat)

                val.append(score)

            data.append(
                {
                    'person': images[x].split('\\')[1].split('_')[0],
                    'compare': images[y].split('\\')[1].split('_')[0],
                    'score': min(val)
                })
    print(poc)
    same = []
    different = []
    print("end processing")
    print("start ploting")
    nulove = 0;
    for person in persons:
        # print(person)
        for item in data:
            p = item['person'].split('_')[0]
            c = item['compare'].split('_')[0]
            # print(c)
            if p == person and c == person:
                # print(item['score'])
                same.append(item['score'])
            elif p == person and c != person:
                different.append(item['score'])
        # print(len(different))
    print(len(different))
    # print(len(same))
    print(nulove)
    print(different)
    # print(same)
    plt.title('Hamming Distance')
    plt.hist(different, bins='auto', label='Different')
    # plt.hist(same,  bins=30, color='y', alpha=0.3, label='Same')
    plt.show()

path_binar = "C:/Users/Erik/PycharmProjects/fei_biom/iris/binarize"
path_mask = "C:/Users/Erik/PycharmProjects/fei_biom/iris/mask_file"


images = load_images(path_binar)
masks = load_images(path_mask)


hammingation(images, masks)


