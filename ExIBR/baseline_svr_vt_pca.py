from sklearn import svm, decomposition
from datetime import datetime
from numpy import genfromtxt
import pickle
import numpy

print 'loading data'

read_file_train = open('AUC_new_dataset_train_811_norm_10000.pkl', 'rb')
read_file_valid = open('AUC_new_dataset_valid_811_norm_10000.pkl', 'rb')
read_file_test = open('AUC_new_dataset_test_811_norm_10000.pkl', 'rb')
read_file_txt_train = open('AUC_new_dataset_unified_text_train8110_10000.pkl', 'rb')
read_file_txt_valid = open('AUC_new_dataset_unified_text_valid8110_10000.pkl', 'rb')
read_file_txt_test = open('AUC_new_dataset_unified_text_test8110_10000.pkl', 'rb')
print 'loading train data'
train_set = numpy.asarray(pickle.load(read_file_train), dtype='float64')
train_txt_set = numpy.asarray(pickle.load(read_file_txt_train), dtype='float64')
print 'loading test data'
test_set = numpy.asarray(pickle.load(read_file_test))
test_txt_set = numpy.asarray(pickle.load(read_file_txt_test))
train_set_size = train_set[0].shape[0]
test_set_size = test_set[0].shape[0]
train_set_xi_v, train_set_xj_v, train_set_xk_v = train_set[0], train_set[1], train_set[2]
train_set_xi_c, train_set_xj_c, train_set_xk_c = train_txt_set[0], train_txt_set[1], train_txt_set[2]
test_set_xi_v, test_set_xj_v, test_set_xk_v = test_set[0], test_set[1], test_set[2]
test_set_xi_c, test_set_xj_c, test_set_xk_c = test_txt_set[0], test_txt_set[1], test_txt_set[2]
test_k_10 = numpy.asarray(numpy.loadtxt('test_k_10_10000.csv'))

print 'loaded data'

def main(pca_n, beta):

    train_data_v = []
    train_data_c = []
    train_label = []

    for i in range(train_set_size):
        a = numpy.append(train_set_xi_v[i], train_set_xj_v[i])
        b = numpy.append(train_set_xi_v[i], train_set_xk_v[i])
        c = numpy.append(train_set_xi_c[i], train_set_xj_c[i])
        d = numpy.append(train_set_xi_c[i], train_set_xk_c[i])
        train_data_v.append(a)
        train_data_v.append(b)
        train_data_c.append(c)
        train_data_c.append(d)
        train_label.append(int(1))
        train_label.append(int(0))


    print str(datetime.now())
    print 'pca start'

    pca_v = decomposition.PCA(n_components=pca_n)
    pca_c = decomposition.PCA(n_components=pca_n)
    train_data_pca_v = pca_v.fit_transform(train_data_v)
    train_data_pca_c = pca_c.fit_transform(train_data_c)

    print str(datetime.now())
    print 'pca ended'


    print str(datetime.now())
    print 'svm train start'

    clf_v = svm.SVR(C=8, cache_size=2000, coef0=0.0, degree=3, gamma=0.125, kernel='rbf', shrinking=True, tol=0.001, verbose=False).fit(train_data_pca_v, train_label)
    clf_c = svm.SVR(C=8, cache_size=2000, coef0=0.0, degree=3, gamma=0.125, kernel='rbf', shrinking=True, tol=0.001,
                    verbose=False).fit(train_data_pca_c, train_label)


    print str(datetime.now())
    print 'svm train ended'

    print str(datetime.now())
    print 'svm test start'

    test_compare = []
    k = 10

    for i in range(test_k_10.shape[0]):

        compare_v = []
        compare_c = []
        for j in range(k):
            v = numpy.append(numpy.asarray(test_set_xi_v[int(test_k_10[i][0])]),
                             numpy.asarray(test_set_xi_v[int(test_k_10[i][j + 1])]))
            c = numpy.append(numpy.asarray(test_set_xi_c[int(test_k_10[i][0])]),
                             numpy.asarray(test_set_xi_c[int(test_k_10[i][j + 1])]))
            compare_v.append(v)
            compare_c.append(c)

        compare_pca_v = pca_v.transform(compare_v)
        y_pred_v = clf_v.predict(compare_pca_v)
        compare_pca_c = pca_c.transform(compare_c)
        y_pred_c = clf_c.predict(compare_pca_c)
        y_pred = (1 - beta) * y_pred_v + beta * y_pred_c

        test_compare.append(y_pred)

    performance = 0.0
    for i in range(test_compare.__len__()):
        count = 0.0
        for j in range(k):
            if test_compare[i][j] <= test_compare[i][0]:
                count = count + 1
        performance = performance + float(1 / count)

    test_score = performance / test_compare.__len__()

    print 'test socre is %f, n is %i, beta is %f'%(test_score, pca_n, beta)
    return test_score

if __name__ == '__main__':
    best_score = 0
    best_n = 10
    best_beta = 0.9
    for n in [5]:
        for beta in [0.4, 0.9, 0.8, 1.0]:
            score = main(n, beta)
            if score>best_score:
                best_score = score
                best_n = n
                best_beta = beta
                print 'best_score is %f, best_n is %i, best_beta is %f'% (best_score, best_n, best_beta)

