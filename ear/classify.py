import matplotlib.pyplot as plt
import numpy as np

import pickle

file_dir = '../cache/ellipse/ear.zip/'
out_dir = '../cache/'

# test_index = 3
feature_type = 'pca'
dist_type = 'm'

global_avg = 120313.0
global_var = 4.85388e+08
global_z_thresh = 20.1877479975


def distance(data_1, data_2, dist="m"):
    if dist == "m":
        return np.sum(np.abs(data_1 - data_2))
    else:
        return np.sqrt(np.sum((data_1 - data_2) ** 2))


# def find_closest(target_label, data, d="m"):
#     test_data = data[target_label][test_index]
#
#     results = []
#     min_val = float("inf")
#     min_label = ''
#     for label in data:
#         for i in range(4):
#             if i == test_index:
#                 continue
#             dist = distance(test_data, data[label][i])
#             results.append(dist)
#             if dist < min_val:
#                 min_val = dist
#                 min_label = label
#
#     z = ((min_val - np.mean(results)) / np.var(results)) * -10000
#
#     return min_label, z


def classify(index_1, index_2, thresh):
    data = pickle.load(open(out_dir + 'ear-' + feature_type + '.pickle', 'rb'))
    label_1 = list(data.keys())[int(index_1 / 4)]
    label_2 = list(data.keys())[int(index_2 / 4)]
    i_1 = index_1 % 4
    i_2 = index_2 % 4

    data_1 = data[label_1][i_1]
    data_2 = data[label_2][i_2]

    dist = distance(data_1, data_2)

    z = ((dist - global_avg) / global_var) * -1000000

    return z, z > thresh, label_1 == label_2


# print(classify('01'))

def test(thresh):
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

            z, predict, same = classify(i, j, thresh)
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


all_tpr = []
all_fpr = []
for thresh in range(-150, 250, 25):
    tpr, fpr = test(thresh)
    all_tpr.append(tpr)
    all_fpr.append(fpr)
    print('thresh', thresh)

plt.plot(all_fpr, all_tpr, '-')
plt.plot([0, 1], [0, 1], '-')

plt.xlabel('True positive rate (Sensitivity)')
plt.ylabel('False positive rate (Specificity)')
plt.title('ROC')
plt.grid(True)
plt.show()
