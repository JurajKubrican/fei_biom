from ear.classify import classify as classify_ear
from ear.classify import show as show_ear
from face.find import normalize as classify_face
from face.find import show_face as show_face
from iris.iris_5 import classify as classify_iris
from iris.iris_5 import print_img as show_iris

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

glob_thresh = 2.0
glob_range = range(-15, 25, 1)


def tp_fp_helper(stats, thresh, same, z):
    predict = z > thresh
    if same:
        if predict:
            stats['tp'] += 1
        else:
            stats['fn'] += 1
    else:
        if predict:
            stats['fp'] += 1
        else:
            stats['tn'] += 1
    return stats


def tp_fp(thresh):
    stats = {
        'ear': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        'face': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        'iris': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        'all': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
    }
    z_same = {
        'ear': [],
        'face': [],
        'iris': [],
        'all': []
    }
    z_diff = {
        'ear': [],
        'face': [],
        'iris': [],
        'all': []
    }
    for i in range(10 * 4):
        for j in range(10 * 4):
            if i == j:
                continue
            z_ear, same = classify_ear(i, j)
            z_face, _ = classify_face(i, j)
            # z_iris, _ = classify_iris(i, j)
            z_iris = 0
            z = np.mean([z_ear, z_face, z_iris])

            stats['ear'] = tp_fp_helper(stats['ear'], thresh, same, z_ear)
            stats['face'] = tp_fp_helper(stats['face'], thresh, same, z_face)
            stats['iris'] = tp_fp_helper(stats['iris'], thresh, same, z_iris)
            stats['all'] = tp_fp_helper(stats['all'], thresh, same, z)

            if same:
                z_same['ear'].append(z_ear)
                z_same['face'].append(z_face)
                z_same['iris'].append(z_iris)
                z_same['all'].append(z)
            else:
                z_diff['ear'].append(z_ear)
                z_diff['face'].append(z_face)
                z_diff['iris'].append(z_iris)
                z_diff['all'].append(z)

    return {
               'ear': np.mean(z_same['ear']),
               'face': np.mean(z_same['face']),
               'iris': np.mean(z_same['iris']),
               'all': np.mean(z_same['all']),
           }, {
               'ear': np.mean(z_diff['ear']),
               'face': np.mean(z_diff['face']),
               'iris': np.mean(z_diff['iris']),
               'all': np.mean(z_diff['all']),
           }, {
               'ear': {'tpr': stats['ear']['tp'] / (stats['ear']['tp'] + stats['ear']['fn']),
                       'fpr': (stats['ear']['tn'] / (stats['ear']['fp'] + stats['ear']['tn']))},
               'face': {'tpr': stats['face']['tp'] / (stats['face']['tp'] + stats['face']['fn']),
                        'fpr': (stats['face']['tn'] / (stats['face']['fp'] + stats['face']['tn']))},
               'iris': {'tpr': stats['iris']['tp'] / (stats['iris']['tp'] + stats['iris']['fn']),
                        'fpr': (stats['iris']['tn'] / (stats['iris']['fp'] + stats['iris']['tn']))},
               'all': {'tpr': stats['all']['tp'] / (stats['all']['tp'] + stats['all']['fn']),
                       'fpr': (stats['all']['tn'] / (stats['all']['fp'] + stats['all']['tn']))},
           }


def show_roc():
    stats_all = []
    for thresh in glob_range:
        print('thresh', thresh)
        z_same, z_diff, stats = tp_fp(thresh)
        print(z_same, z_diff)
        stats_all.append(stats)

    all_fpr = {'ear': [],
               'face': [],
               'iris': [],
               'all': []}
    all_tpr = {'ear': [],
               'face': [],
               'iris': [],
               'all': []}

    for item in stats_all:
        all_fpr['ear'].append(item['ear']['fpr'])
        all_fpr['face'].append(item['face']['fpr'])
        all_fpr['iris'].append(item['iris']['fpr'])
        all_fpr['all'].append(item['all']['fpr'])

        all_tpr['ear'].append(item['ear']['tpr'])
        all_tpr['face'].append(item['face']['tpr'])
        all_tpr['iris'].append(item['iris']['tpr'])
        all_tpr['all'].append(item['all']['tpr'])

    blue_patch = mpatches.Patch(color='blue', label='Iris')
    red_patch = mpatches.Patch(color='red', label='Ear')
    green_patch = mpatches.Patch(color='green', label='Face')
    yellow_patch = mpatches.Patch(color='yellow', label='All')

    plt.plot([1 - x for x in all_fpr['iris']], all_tpr['iris'], '-b')
    plt.plot([1 - x for x in all_fpr['ear']], all_tpr['ear'], '-r')
    plt.plot([1 - x for x in all_fpr['face']], all_tpr['face'], '-g')
    plt.plot([1 - x for x in all_fpr['all']], all_tpr['all'], '-y')
    plt.plot([0, 1], [0, 1], '-')
    plt.xlabel('False positive rate (Specificity)')
    plt.ylabel('True positive rate (Sensitivity)')
    plt.title('ROC')
    plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch])
    plt.grid(True)
    plt.show(1)

    red_patch = mpatches.Patch(color='red', label='Fpr')
    green_patch = mpatches.Patch(color='green', label='Tpr')

    plt.xlabel('Threshold')
    plt.ylabel('Tpr, Fpr')
    plt.plot(glob_range, all_tpr['iris'], '-g')
    plt.plot(glob_range, all_fpr['iris'], '-r')
    plt.title('TPR FPR Iris')
    plt.legend(handles=[red_patch, green_patch])
    plt.grid(True)
    plt.show(2)

    plt.xlabel('Threshold')
    plt.ylabel('Tpr, Fpr Ear')
    plt.plot(glob_range, all_tpr['ear'], '-g')
    plt.plot(glob_range, all_fpr['ear'], '-r')
    plt.title('TPR FPR')
    plt.legend(handles=[red_patch, green_patch])
    plt.grid(True)
    plt.show(3)

    plt.xlabel('Threshold')
    plt.ylabel('Tpr, Fpr Face')
    plt.plot(glob_range, all_tpr['face'], '-g')
    plt.plot(glob_range, all_fpr['face'], '-r')
    plt.title('TPR FPR')
    plt.legend(handles=[red_patch, green_patch])
    plt.grid(True)
    plt.show(4)

    plt.xlabel('Threshold')
    plt.ylabel('Tpr, Fpr All')
    plt.plot(glob_range, all_tpr['all'], '-g')
    plt.plot(glob_range, all_fpr['all'], '-r')
    plt.title('ROC')
    plt.legend(handles=[red_patch, green_patch])
    plt.grid(True)
    plt.show(5)


show_roc()


def compare(x, y):
    z_ear, same = classify_ear(x, y)
    z_face, _ = classify_face(x, x)
    # z_iris, _ = classify_iris(i, j)
    z_iris = 0
    z = np.mean([z_ear, z_face, z_iris])

    print(glob_thresh > z)

    show_ear(x)
    show_ear(y)
    show_face(x)
    show_face(y)
    show_iris(x, y)

    return 0

# compare(1, 10)
