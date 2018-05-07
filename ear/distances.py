import numpy as np

import pickle

file_dir = '../cache/ellipse/ear.zip/'
out_dir = '../cache/'
lbp_file = out_dir + 'ear-lbp.pickle'
pca_file = out_dir + 'ear-pca.pickle'

lbp_data = pickle.load(open(lbp_file, 'rb'))
pca_data = pickle.load(open(pca_file, 'rb'))


# print(lbp_data)


def distances(ear_1, ear_2):
    lbp_data_1 = np.asarray(lbp_data['data'][ear_1])
    lbp_data_2 = np.asarray(lbp_data['data'][ear_2])

    dist_lbp_e = np.sqrt(np.sum((lbp_data_1 - lbp_data_2) ** 2))
    dist_lbp_m = np.sum(np.abs(lbp_data_1 - lbp_data_2))

    pca_data_1 = np.asarray(pca_data['data'][ear_1])
    pca_data_2 = np.asarray(pca_data['data'][ear_2])

    dist_pca_e = np.sqrt(np.sum((pca_data_1 - pca_data_2) ** 2))
    dist_pca_m = np.sum(np.abs(pca_data_1 - pca_data_2))

    return dist_lbp_e, dist_lbp_m, dist_pca_e, dist_pca_m


thresholds = (0.4, 1.5, 1900, 1900)
accurate = ()

avg_same = []
avg_different = []
results = [0, 0, 0, 0]
for i in range(len(lbp_data['labels'])):
    for j in range(len(lbp_data['labels'])):
        if (i == j):
            continue
        label_1 = lbp_data['labels'][i]
        label_2 = lbp_data['labels'][j]
        dist_lbp_e, dist_lbp_m, dist_pca_e, dist_pca_m = distances(i, j)

        if label_1 == label_2:
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
print(np.average([x[0] for x in avg_same]))
print(np.average([x[0] for x in avg_different]))
print((results[0] / (len(avg_same[:]) + len(avg_different[:]))) * 100)

# bins = [0, 0.150]
# hist, bin_edges = np.histogram([x[0] for x in avg_same], bins=bins)

print('mahattan LBP')
print(np.average([x[1] for x in avg_same]))
print(np.average([x[1] for x in avg_different]))
print((results[1] / (len(avg_same[:]) + len(avg_different[:]))) * 100)

print('euclid PCA')
print(np.average([x[2] for x in avg_same]))
print(np.average([x[2] for x in avg_different]))
print((results[2] / (len(avg_same[:]) + len(avg_different[:]))) * 100)

print('mahattan PCA')
print(np.average([x[3] for x in avg_same]))
print(np.average([x[3] for x in avg_different]))
print((results[3] / (len(avg_same[:]) + len(avg_different[:]))) * 100)
