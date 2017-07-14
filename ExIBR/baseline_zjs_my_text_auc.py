import os
import sys
import time
import numpy
from rank_metrics import ndcg_at_k
import gzip
#import cPickle
import pickle
from  datetime  import  *

read_file_valid = open('AUC_new_dataset_unified_text_valid8110.pkl', 'rb')
read_file_test = open('AUC_new_dataset_unified_text_test8110.pkl', 'rb')

print 'loading text valid data'
print ('now():'+str( datetime.now() ))
valid_set = numpy.asarray(pickle.load(read_file_valid))

print 'loading text test data'
print ('now():'+str( datetime.now() ))
test_set = numpy.asarray(pickle.load(read_file_test))


valid_set_xi, valid_set_xj, valid_set_xk = valid_set[0], valid_set[1], valid_set[2]
test_set_xi, test_set_xj, test_set_xk = test_set[0], test_set[1], test_set[2]

valid_set_size = valid_set[0].shape[0]
test_set_size = test_set[0].shape[0]

def valid_model(size):
    vi = numpy.array(valid_set_xi)
    vj = numpy.transpose(numpy.array(valid_set_xj))
    vk = numpy.transpose(numpy.array(valid_set_xk))
    vij = numpy.dot(vi, vj)
    vik = numpy.dot(vi, vk)
    top = []
    performance = 0.0
    l = vi.tolist()
    for i in range(size):
        if (l[i] not in top):
            top.append(l[i])
    for i in range(top.__len__()):
        t = top[i]
        count1 = 0.0
        count2 = 0.0
        for j in range(size):
            if (l[j] == t):
                count1 = count1 + 1
                sup = (vij[j][j] - vik[j][j])
                if (sup > 0):
                    count2 = count2 + 1
        performance = performance + float(count2 / count1)

    return float(performance / top.__len__())


def test_model(size):
    ti = numpy.array(test_set_xi)
    tj = numpy.transpose(numpy.array(test_set_xj))
    tk = numpy.transpose(numpy.array(test_set_xk))
    tij = numpy.dot(ti, tj)
    tik = numpy.dot(ti, tk)
    top = []
    performance = 0.0
    l = ti.tolist()
    for i in range(size):
        if (l[i] not in top):
            top.append(l[i])
    for i in range(top.__len__()):
        t = top[i]
        count1 = 0.0
        count2 = 0.0
        for j in range(size):
            if (l[j] == t):
                count1 = count1 + 1
                sup = (tij[j][j] - tik[j][j])
                if (sup > 0):
                    count2 = count2 + 1
        performance = performance + float(count2 / count1)

    return float(performance / top.__len__())

def baseline():

    valid = valid_model(valid_set_size)
    test = test_model(test_set_size)
    print valid
    print test

if __name__ == '__main__':
    baseline()
    print valid_set_size
    print test_set_size