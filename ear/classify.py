import numpy as np

import pickle

file_dir = '../cache/ellipse/ear.zip/'
out_dir = '../cache/'

test_index = 3
feature_type = 'pca'
dist_type = 'e'


def distance(data_1, data_2, dist="m"):
    if dist == "m":
        return np.sum(np.abs(data_1 - data_2))
    else:
        return np.sqrt(np.sum((data_1 - data_2) ** 2))


def find_closest(target_label, data, d="m"):
    test_data = data[target_label][test_index]

    results = []
    min_val = float("inf")
    min_label = ''
    for label in data:
        for i in range(4):
            if i == test_index:
                continue
            dist = distance(test_data, data[label][i])
            results.append(dist)
            if dist < min_val:
                min_val = dist
                min_label = label

    z = ((min_val - np.mean(results)) / np.var(results)) * -10000

    return min_label, z


def classify(label):
    data = pickle.load(open(out_dir + 'ear-' + feature_type + '.pickle', 'rb'))
    closest, z = find_closest(label, data, dist_type)

    return closest, z


# print(classify('01'))

def test():
    data = pickle.load(open(out_dir + 'ear-' + feature_type + '.pickle', 'rb'))
    pos = []
    neg = []
    for label in data:
        result, z = classify(label)
        if result == label:
            print(result, " z: ", z)
            pos.append(z)
        else:
            print(label, " -> ", result, ' z: ', z)
            neg.append(z)

    print(len(pos), np.average(pos))
    print(len(neg), np.average(neg))
    print(len(pos) / len(neg) + len(pos))


test()
