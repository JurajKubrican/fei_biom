from pathlib import Path

import numpy as np

from os import listdir
import pickle

file_dir: str = '../cache/ellipse/ear.zip/'
out_dir: str = '../cache/'
lbp_file: str = out_dir + 'ear-lbp.pickle'
pca_file: str = out_dir + 'ear-pca.pickle'

lbp_data = pickle.load(open(lbp_file, 'rb'))
pca_data = pickle.load(open(lbp_file, 'rb'))


# print(lbp_data)


def distances(ear_1, ear_2):
    if ear_1 not in pca_data:
        return False, False, False, False

    lbp_data_1 = np.asarray(lbp_data[ear_1])
    lbp_data_2 = np.asarray(lbp_data[ear_2])

    dist_lbp_e = np.sqrt(np.sum((lbp_data_1 - lbp_data_2) ** 2))
    dist_lbp_m = np.sum(np.abs(lbp_data_1 - lbp_data_2))

    pca_data_1 = np.asarray(pca_data[ear_1])
    pca_data_2 = np.asarray(pca_data[ear_2])

    dist_pca_e = np.sqrt(np.sum((pca_data_1 - pca_data_2) ** 2))
    dist_pca_m = np.sum(np.abs(pca_data_1 - pca_data_2))

    return dist_lbp_e, dist_lbp_m, dist_pca_e, dist_pca_m


ear_1 = '58-1.BMP'
ear_2 = '58-2.BMP'
ear_3 = '58-3.BMP'
ear_4 = '57-1.BMP'
thresholds = (1.0, 0.5, 0.56, 1.0)
accurate = ()

avg_same = []
avg_different = []
results = [0, 0, 0, 0]
for label_1 in lbp_data:
    for label_2 in lbp_data:
        if (label_1 == label_2):
            continue
        dist_lbp_e, dist_lbp_m, dist_pca_e, dist_pca_m = distances(label_1, label_2)

        if label_1[:2] == label_2[:2]:
            if dist_lbp_e < thresholds[0]:
                results[0] += 1

            if dist_lbp_m < thresholds[1]:
                results[1] += 1

            if dist_pca_e < thresholds[2]:
                results[2] += 1

            if dist_pca_m < thresholds[3]:
                results[3] += 1

            avg_same.append((dist_lbp_e, dist_lbp_m, dist_pca_e, dist_pca_m))
        else:
            if dist_lbp_e >= thresholds[0]:
                results[0] += 1

            if dist_lbp_m >= thresholds[1]:
                results[1] += 1

            if dist_pca_e >= thresholds[2]:
                results[2] += 1

            if dist_pca_m >= thresholds[3]:
                results[3] += 1
            avg_different.append((dist_lbp_e, dist_lbp_m, dist_pca_e, dist_pca_m))

print('euclid LBP')
print(np.average(avg_same[:][0]))
print(np.average(avg_different[:][0]))
print((results[0] / (len(avg_same) + len(avg_different))) * 100)

print('mahattan LBP')
print(np.average(avg_same[:][1]))
print(np.average(avg_different[:][1]))
print((results[1] / (len(avg_same) + len(avg_different))) * 100)

print('euclid PCA')
print(np.average(avg_same[:][2]))
print(np.average(avg_different[:][2]))
print((results[2] / (len(avg_same) + len(avg_different))) * 100)

print('mahattan PCA')
print(np.average(avg_same[:][3]))
print(np.average(avg_different[:][3]))
print((results[3] / (len(avg_same) + len(avg_different))) * 100)
