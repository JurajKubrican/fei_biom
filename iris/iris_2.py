import cv2

import os
import os.path
import decimal
import math
from matplotlib import pyplot as plt
import numpy as np

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def load_images(path):
    images = []
    for root,dirs,filename in os.walk(path):
        for image in filename:
            if os.path.splitext(image)[1] == ".bmp":
                images.append(os.path.join(root, image))
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



def processing(path,coord):
    img = cv2.imread(path)
    radiusall = int(coord[6])       #adius velkeho kruhu
    radiuspupil = int(coord[3])     #radius maleho kruhu
    radiusdown = int(coord[9])
    radiusup = int(coord[12])
    center_p = (int(coord[1]), int(coord[2]))
    center_i = (int(coord[4]), int(coord[5]))
    center_down = (int(coord[7]), int(coord[8]))
    center_up = (int(coord[10]), int(coord[11]))

    iris_radius = +60          #radiusall - radiuspupil
    nsamples = 360
    samples = np.linspace(0, 2.0 * np.pi, nsamples)[:-1]
    polar = np.zeros((iris_radius, int(nsamples)))
    mask = np.zeros((iris_radius, int(nsamples)))
    margin = np.zeros((iris_radius, 6))
    margin += 255
    step = 1 / iris_radius

    for theta in samples:
        for r in drange(0, 1, step):
            xp = np.cos(theta) * radiuspupil + center_p[0]
            yp = abs(np.sin(theta) * radiuspupil - center_p[1])
            xs = np.cos(theta) * radiusall + center_i[0]
            ys = abs(np.sin(theta) * radiusall - center_i[1])
            x = int(round((1 - r) * xp + r * xs))
            y = int(round((1 - r) * yp + r * ys))

            point = np.array([x, y])
            distdown = np.linalg.norm(point - np.array(center_down))
            distup = np.linalg.norm(point - np.array(center_up))
            index = int(round(r * iris_radius))

            if index >= iris_radius:
                index -= 1
            polar[index][int(np.degrees(theta))] = img[y][x][0]
            if distdown > radiusdown or distup > radiusup:
                mask[index][int(np.degrees(theta))] = 255

    # filenamep = "C:/Users/Erik/PycharmProjects/fei_biom/iris/polar/" + coord[0]
    # print(coord[0].split('/')[2])
    # filenamem = "C:/Users/Erik/PycharmProjects/fei_biom/iris/mask/" + coord[0]
    # filename_out = "C:/Users/Erik/PycharmProjects/fei_biom/iris/processing/" + coord[0]
    output = np.concatenate((margin, polar, margin, mask, margin), axis=1)

    filename_out = "C:/Users/Erik/PycharmProjects/fei_biom/iris/mask_file/" + coord[0].split('/')[2]
    # print(filename_out)
    # cv2.imwrite(filename_out, output)
    cv2.imwrite(filename_out, mask)
    # cv2.imwrite(filenamep, polar)
    # cv2.imwrite(filenamem, mask)





    return polar, mask


path = "C:/Users/Erik/PycharmProjects/fei_biom/cache/extract/iris/"
surr_path = "./iris_bounding_circles.csv"

images = load_images(path)
coordinates = load_coordinates(surr_path)
igg = plt.imread(images[0])

x=0;

for image in images:
    polar, mask = processing(image,coordinates[x])
    x+=1;


# polar, mask = processing(images[0],coordinates[x])

# plt.imshow(polar)
# plt.interactive(False)
# plt.show(block = True)
