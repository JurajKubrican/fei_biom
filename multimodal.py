from ear.classify import classify as classify_ear
from face.find import normalize as classify_face

import matplotlib.pyplot as plt
import numpy as np


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

            # z_ear, same = classify_ear(i, j)
            z_face, same = classify_face(i, j)

            # z = np.mean([z_ear, z_face])
            z = z_face
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

    return tp / (tp + fn), 1 - (tn / (fp + tn))


all_tpr = []
all_fpr = []
for thresh in range(-150, 250, 1):
    tpr, fpr = tp_fp(thresh)
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
