import os
import sys
import time
import numpy
from rank_metrics import ndcg_at_k
import gzip
#import cPickle
import pickle
from  datetime  import  *

print 'loading visual valid data'
print ('now():'+str( datetime.now() ))
read_file_valid = open('AUC_new_dataset_valid0.pkl', 'rb')
print 'loading visual test data'
print ('now():'+str( datetime.now() ))
read_file_test = open('AUC_new_dataset_test0.pkl', 'rb')


valid_set = numpy.asarray(pickle.load(read_file_valid))
test_set = numpy.asarray(pickle.load(read_file_test))


valid_set_xi, valid_set_xj, valid_set_xk = valid_set[0], valid_set[1], valid_set[2]
test_set_xi, test_set_xj, test_set_xk = test_set[0], test_set[1], test_set[2]

valid_set_size = valid_set[0].shape[0]
test_set_size = test_set[0].shape[0]

def valid_model(size):
    vi = numpy.array(valid_set_xi)
    vj = numpy.transpose(numpy.array(valid_set_xj))
    vk = numpy.transpose(numpy.array(valid_set_xk))
    count = 0.0
    vij = numpy.dot(vi, vj)
    vik = numpy.dot(vi, vk)
    for i in range(size):
        ij = vij[i][i]
        ik = vik[i][i]
        if (ij > ik):
            count = count + 1

    performance = float(count / size)

    return performance


def test_model(size):
    ti = numpy.array(test_set_xi)
    tj = numpy.transpose(numpy.array(test_set_xj))
    tk = numpy.transpose(numpy.array(test_set_xk))
    count = 0.0
    tij = numpy.dot(ti, tj)
    tik = numpy.dot(ti, tk)
    for i in range(size):
        ij = tij[i][i]
        ik = tik[i][i]
        if (ij > ik):
            count = count + 1

    performance = float(count / size)

    return performance

def baseline():

    valid = valid_model(valid_set_size)
    test = test_model(test_set_size)
    print valid
    print test

if __name__ == '__main__':
    baseline()
    print valid_set_size
    print test_set_size