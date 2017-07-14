from sklearn import svm, decomposition

from datetime import datetime
from numpy import genfromtxt
import pickle
import numpy


def main(pca_n):
    print 'loading data'

    read_file_train = open('AUC_new_dataset_train_811_norm_10000.pkl', 'rb')
    read_file_valid = open('AUC_new_dataset_valid_811_norm_10000.pkl', 'rb')
    read_file_test = open('AUC_new_dataset_test_811_norm_10000.pkl', 'rb')

    train_set = numpy.asarray(pickle.load(read_file_train))
    valid_set = numpy.asarray(pickle.load(read_file_valid))
    test_set = numpy.asarray(pickle.load(read_file_test))

    train_set_size = train_set[0].shape[0]
    valid_set_size = valid_set[0].shape[0]
    test_set_size = test_set[0].shape[0]

    train_set_xi, train_set_xj, train_set_xk = train_set[0], train_set[1], train_set[2]
    valid_set_xi, valid_set_xj, valid_set_xk = valid_set[0], valid_set[1], valid_set[2]
    test_set_xi, test_set_xj, test_set_xk = test_set[0], test_set[1], test_set[2]

    test_k_10 = numpy.asarray(numpy.loadtxt('test_k_10_10000.csv'))

    print 'loaded data'

    train_data = []
    train_label = []

    for i in range(train_set_size):
        a = numpy.append(train_set_xi[i], train_set_xj[i])
        b = numpy.append(train_set_xi[i], train_set_xk[i])
        train_data.append(a)
        train_data.append(b)
        train_label.append(int(1))
        train_label.append(int(0))

    print str(datetime.now())
    print 'pca start'

    pca = decomposition.PCA(n_components=pca_n)
    train_data_pca = pca.fit_transform(train_data)
    print train_data_pca.shape

    print str(datetime.now())
    print 'pca ended'

    print str(datetime.now())
    print 'svm train start'

    clf = svm.SVR(C=8, cache_size=2000, coef0=0.0, degree=3, gamma=0.125, kernel='rbf', shrinking=True, tol=0.001,
                  verbose=False).fit(train_data_pca, train_label)

    print str(datetime.now())
    print 'svm train ended'

    print str(datetime.now())
    print 'svm test start'

    test_compare = []
    k = 10

    for i in range(test_k_10.shape[0]):

        compare = []
        for j in range(k):
            c = numpy.append(numpy.asarray(test_set_xi[int(test_k_10[i][0])]),
                             numpy.asarray(test_set_xi[int(test_k_10[i][j + 1])]))
            compare.append(c)

        compare_pca = pca.transform(compare)
        y_pred = clf.predict(compare_pca)
        test_compare.append(y_pred)

    performance = 0.0
    for i in range(test_compare.__len__()):
        count = 0.0
        for j in range(k):
            if test_compare[i][j] <= test_compare[i][0]:
                count = count + 1
        performance = performance + float(1 / count)

    test_score = performance / test_compare.__len__()

    print 'test socre is %f' % (test_score)
    return test_score


if __name__ == '__main__':
    best_score = 0
    best_n = 10
    for n in [5, 100, 200, 500]:

        score = main(n)
        if score > best_score:
            best_score = score
            best_n = n
    print 'best_score is %f, best_n is %i' % (best_score, best_n)
