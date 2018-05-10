import numpy as np

import pickle

file_dir = '../cache/ellipse/ear.zip/'
out_dir = '../cache/'
lbp_file = out_dir + 'ear-lbp.pickle'
pca_file = out_dir + 'ear-pca.pickle'

lbp_data = pickle.load(open(lbp_file, 'rb'))
pca_data = pickle.load(open(pca_file, 'rb'))


# print(lbp_data)


def distance(data_1, data_2, dist="m"):
    if dist == "m":
        return np.sum(np.abs(data_1 - data_2))
    else:
        return np.sqrt(np.sum((data_1 - data_2) ** 2))


thresholds = (0.4, 1.5, 1900, 1900)
accurate = ()

results = []
avg_different = []
for label_1 in pca_data:
    for i in range(4):
        for label_2 in pca_data:
            for j in range(4):
                if label_1 == label_2 and i == j:
                    continue

                data_1 = pca_data[label_1][i]
                data_2 = pca_data[label_2][j]

                dist = distance(data_1, data_2)

                results.append(dist)

print(np.mean(results))
print(np.var(results, ddof=1))
