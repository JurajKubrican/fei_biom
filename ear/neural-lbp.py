import numpy as np
import pickle

from sklearn.neural_network import MLPClassifier

from sklearn import svm

depth = 255.0
source_dir = ''

cahce_dir = '../cache/'
source = cahce_dir + 'ear-lbp.pickle'
neural_cache = cahce_dir + 'ear-lbp-nerual.pickle'

output_dir = 'dataset/'

# must add to 1
part = 0.3
train = 0.8 * part
test = 0.1 * part
valid = 0.1 * part

numlabels = 0

diff = 0.01


def reformat_data():
    pca_data = pickle.load(open(source, 'rb'))

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    label = ''
    numInLabel = 0
    for file in pca_data:
        print(file)
        if (label != file[:2]):
            label = file[:2]
            numInLabel = 0

        numInLabel += 1

        if numInLabel < 3:
            train_data.append(np.asarray(pca_data[file]).flatten())
            train_labels.append(label)
        else:
            test_data.append(np.asarray(pca_data[file]).flatten())
            test_labels.append(label)

    all_pickle = {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
    }

    print('dumping result to file.')
    pickle.dump(all_pickle, open(neural_cache, 'wb'))


reformat_data()


def mlp(layers):
    data = pickle.load(open(neural_cache, 'rb'))
    train_data = data['train_data']
    train_labels = data['train_labels']
    test_data = data['test_data']
    test_labels = data['test_labels']

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=layers, random_state=1)
    clf.fit(train_data, train_labels)
    # MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
    #               beta_1=0.9, beta_2=0.999, early_stopping=False,
    #               epsilon=1e-08, hidden_layer_sizes=(30, 30), learning_rate='constant',
    #               learning_rate_init=0.001, max_iter=200, momentum=0.9,
    #               nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
    #               solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
    #               warm_start=False)

    print('predicting...')

    labels = clf.predict(test_data)
    correct = np.count_nonzero(np.asarray(labels) == np.asarray(test_labels))

    print(correct / len(labels) * 100)


mlp((303, 200, 200, 200, 100, 100, 100))


def my_svm(kernel):
    data = pickle.load(open(neural_cache, 'rb'))
    train_data = data['train_data']
    train_labels = data['train_labels']
    test_data = data['test_data']
    test_labels = data['test_labels']

    clf = svm.SVC(kernel=kernel)
    clf.fit(train_data, train_labels)
    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)

    print('predicting...')

    labels = clf.predict(test_data)
    correct = np.count_nonzero(np.asarray(labels) == np.asarray(test_labels))

    print(correct / len(labels) * 100)


my_svm('rbf')
my_svm('linear')
my_svm('sigmoid')
