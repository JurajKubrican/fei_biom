from pathlib import Path

import numpy as np

from os import listdir
import pickle

file_dir: str = '../cache/ellipse/ear.zip/'
out_dir: str = '../cache/'
lbp_file: str = out_dir + 'ear-lbp.pickle'

lbp_data = pickle.load(open(lbp_file, 'rb'))


# print(lbp_data)


def distances(ear_1, ear_2):
    lbp_data_1 = np.asarray(lbp_data[ear_1])
    lbp_data_2 = np.asarray(lbp_data[ear_2])

    dist_e = np.sqrt(np.sum((lbp_data_1 - lbp_data_2) ** 2))

    dist_m = 6

    print(dist_e)
    return dist_e


ear_1 = '58-1.BMP'
ear_2 = '58-2.BMP'
ear_3 = '58-3.BMP'
ear_4 = '57-1.BMP'
distances(ear_1, ear_1)
distances(ear_1, ear_2)
distances(ear_2, ear_3)
distances(ear_3, ear_4)
