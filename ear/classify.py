import matplotlib.pyplot as plt
import numpy as np
import os

import pickle

file_dir = '../cache/ellipse/ear.zip/'
pickle_dir = os.path.dirname(os.path.abspath(__file__)) + '/../cache/'

# test_index = 3
feature_type = 'pca'
dist_type = 'm'

global_avg = 120313.0
global_var = 4.85388e+08
global_z_thresh = 2.01877479975


def distance(data_1, data_2, dist="m"):
    if dist == "m":
        return np.sum(np.abs(data_1 - data_2))
    else:
        return np.sqrt(np.sum((data_1 - data_2) ** 2))


def classify(index_1, index_2):
    data = pickle.load(open(pickle_dir + 'ear-' + feature_type + '.pickle', 'rb'))
    label_1 = list(data.keys())[int(index_1 / 4)]
    label_2 = list(data.keys())[int(index_2 / 4)]
    i_1 = index_1 % 4
    i_2 = index_2 % 4

    data_1 = data[label_1][i_1]
    data_2 = data[label_2][i_2]

    dist = distance(data_1, data_2)

    z = ((dist - global_avg) / global_var) * -100000

    return z, label_1 == label_2


# print(classify('01'))

def tp_fp(thresh):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    z_same = []
    z_diff = []
    for i in range(10 * 4):
        for j in range(10 * 4):
            if i == j:
                continue

            z, same = classify(i, j)
            predict = z > thresh
            if same:
                z_same.append(z)
                if predict:
                    tp += 1
                else:
                    fn += 1
            else:
                z_diff.append(z)
                if predict:
                    fp += 1
                else:
                    tn += 1

    # print('same', np.median(z_same))
    # print('diff', np.median(z_diff))
    # print('thresh', np.mean([np.median(z_same), np.median(z_diff)]))
    #
    # print('tpr', tp / (tp + fp))
    # print('fpr', 1 - (fp / (tn + fn)))
    return tp / (tp + fn), 1 - (tn / (fp + tn))


def roc():
    all_tpr = []
    all_fpr = []
    for thresh in range(-15, 25, 1):
        tpr, fpr = tp_fp(thresh)
        all_tpr.append(tpr)
        all_fpr.append(fpr)
        print('thresh', thresh)
    return all_fpr, all_tpr


# all_fpr, all_tpr = roc()
#
# plt.plot(all_fpr, all_tpr, '-')
# plt.plot([0, 1], [0, 1], '-')
#
# plt.xlabel('True positive rate (Sensitivity)')
# plt.ylabel('False positive rate (Specificity)')
# plt.title('ROC')
# plt.grid(True)
# plt.show()
