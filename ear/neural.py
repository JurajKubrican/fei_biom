import numpy as np
import pickle

from sklearn.neural_network import MLPClassifier

from sklearn import svm

depth = 255.0
source_dir = ''

cahce_dir = '../cache/'
source_pca = cahce_dir + 'ear-pca.pickle'

output_dir = 'dataset/'

# must add to 1
part = 0.3
train = 0.8 * part
test = 0.1 * part
valid = 0.1 * part

numlabels = 0

diff = 0.01


def reformat_data():
    pca_data = pickle.load(open(source_pca, 'rw'))

    for label in pca_data:
        print(label)

    all_pickle = {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'valid_data': valid_data,
        'valid_labels': valid_labels,
    }

    print('dumping result to file.')
    pickle.dump(all_pickle, open(output_dir + source_dir.replace('/', '') + '-' + str(part) + '.pickle', 'wb'))

reformat_data()


def mlp(numData, layers):
    # data = pickle.load(open(output_dir + source_dir.replace('/', '') + '-' + str(part) + '.pickle', 'rb'))
    data = pickle.load(open(output_dir + source_dir.replace('/', '') + '-intersect-' + str(part) + '.pickle', 'rb'))
    train_data = data['train_data'][0:numData]
    train_labels = data['train_labels'][0:numData]
    test_data = data['test_data']
    test_labels = data['test_labels']
    img_dims = len(train_data[0])
    train_data = np.reshape(train_data, [len(train_data), img_dims * img_dims])
    test_data = np.reshape(test_data, [len(test_data), img_dims * img_dims])

    print('MLP ' + str(layers) + ' ' + str(numData))
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
    correct = np.count_nonzero(labels == test_labels)

    print(correct / len(labels))


# mlp(50, 784)
# mlp(100, 784)
# mlp(1000, 784)
# mlp(5000, 784)
# mlp(20000, 784)

# # mlp(50, (784, 784))
# mlp(100, (784, 784))
# mlp(1000, (784, 784))
# # mlp(5000, (784, 784))
# mlp(20000, (784, 784))
#

mlp(50, 12)
mlp(100, 12)
mlp(1000, 12)
mlp(5000, 12)
mlp(20000, 12)


#
# mlp(50, 50)
# mlp(100, 50)
# mlp(1000, 50)
# mlp(5000, 50)
# mlp(20000, 50)
#
# mlp(50, 100)
# mlp(100, 100)
# mlp(1000, 100)
# mlp(5000, 100)
# mlp(20000, 100)
#
# mlp(50, (30, 30))
# mlp(100, (30, 30))
# mlp(1000, (30, 30))
# mlp(5000, (30, 30))
# mlp(20000, (30, 30))
#
# mlp(50, (50, 50))
# mlp(100, (50, 50))
# mlp(1000, (50, 50))
# mlp(5000, (50, 50))
# mlp(20000, (50, 50))
#
# mlp(50, (100, 100))
# mlp(100, (100, 100))
# mlp(1000, (100, 100))
# mlp(5000, (100, 100))
# mlp(20000, (100, 100))


def my_svm(numData, kernel):
    # data = pickle.load(open(output_dir + source_dir.replace('/', '') + '-' + str(part) + '.pickle', 'rb'))
    data = pickle.load(open(output_dir + source_dir.replace('/', '') + '-intersect-' + str(part) + '.pickle', 'rb'))

    train_data = data['train_data'][0:numData]
    train_labels = data['train_labels'][0:numData]
    test_data = data['test_data']
    test_labels = data['test_labels']
    img_dims = len(train_data[0])
    train_data = np.reshape(train_data, [len(train_data), img_dims * img_dims])
    test_data = np.reshape(test_data, [len(test_data), img_dims * img_dims])

    print('SVM ' + kernel + ' ' + str(numData))

    clf = svm.SVC(kernel=kernel)
    # clf = svm.SVC(kernel='linear')
    # clf = svm.SVC(kernel='sigmoid')
    clf.fit(train_data, train_labels)
    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)

    print('predicting...')

    labels = clf.predict(test_data)
    correct = np.count_nonzero(labels == test_labels)

    print(correct / len(labels))

# # my_svm(50, 'linear')
# my_svm(100, 'rbf')
# my_svm(1000, 'rbf')
# # my_svm(5000, 'linear')
# my_svm(20000, 'rbf')
#
# # my_svm(50, 'linear')
# my_svm(100, 'linear')
# my_svm(1000, 'linear')
# # my_svm(5000, 'linear')
# my_svm(20000, 'linear')
#
# my_svm(50, 'sigmoid')
# my_svm(100, 'sigmoid')
# my_svm(1000, 'sigmoid')
# my_svm(5000, 'sigmoid')
# my_svm(20000, 'sigmoid')
