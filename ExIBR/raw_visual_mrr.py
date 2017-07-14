import os
import sys
import time
import numpy
import gzip
#import cPickle
import pickle
from  datetime  import  *
import random

print 'loading visual valid data'
print ('now():'+str( datetime.now() ))
read_file_valid = open('AUC_new_dataset_valid_811_norm.pkl', 'rb')
print 'loading visual test data'
print ('now():'+str( datetime.now() ))
read_file_test = open('AUC_new_dataset_test_811_norm.pkl', 'rb')


valid_set = numpy.asarray(pickle.load(read_file_valid))
test_set = numpy.asarray(pickle.load(read_file_test))


valid_set_xi, valid_set_xj, valid_set_xk = valid_set[0], valid_set[1], valid_set[2]
test_set_xi, test_set_xj, test_set_xk = test_set[0], test_set[1], test_set[2]

valid_set_size = valid_set[0].shape[0]
test_set_size = test_set[0].shape[0]

def valid_model(size, k):
    vi = numpy.array(valid_set_xi)
    vj = numpy.transpose(numpy.array(valid_set_xj))
    vk = numpy.transpose(numpy.array(valid_set_xk))
    vij = numpy.dot(vi, vj)
    vik = numpy.dot(vi, vk)
    top = []
    performance = 0.0
    vi_list = vi.tolist()
    vk_list = vk.tolist()
    for i in range(size):
        if (vi_list[i] not in top):
            top.append(vi_list[i])
    for i in range(top.__len__()):
        compare = []
        positive = []
        positive_v = []
        t_v = top[i]
        for j in range(size):
            if (vi_list[j] == t_v):
                positive.append(j)
                positive_v.append(vj[j].tolist())
        ij = vij[i][positive[0]]
        compare.append(ij)
        k_list = random.sample(range(size), 3*k)
        for j in k_list:
            if (vk_list[j] not in positive_v):
                ik = vik[i][j]
                compare.append(ik)
            if (compare.__len__() == k):
                break
        count = 0.0
        for j in range(k):
            if (compare[j] <= compare[0]):
                count = count + 1

        performance = performance + float(1 / count)

    return float(performance / top.__len__())


def test_model(size, k):
    ti = numpy.array(test_set_xi)
    tj = numpy.transpose(numpy.array(test_set_xj))
    tk = numpy.transpose(numpy.array(test_set_xk))
    tij = numpy.dot(ti, tj)
    tik = numpy.dot(ti, tk)
    top = []
    performance = 0.0
    ti_list = ti.tolist()
    tk_list = tk.tolist()
    for i in range(size):
        if (ti_list[i] not in top):
            top.append(ti_list[i])
    for i in range(top.__len__()):
        compare = []
        positive = []
        positive_v = []
        t_v = top[i]
        for j in range(size):
            if (ti_list[j] == t_v):
                positive.append(j)
                positive_v.append(tj[j].tolist())
        ij = tij[i][positive[0]]
        compare.append(ij)
        k_list = random.sample(range(size), 3*k)
        for j in k_list:
            if (tk_list[j] not in positive_v):
                ik = tik[i][j]
                compare.append(ik)
            if (compare.__len__() == k):
                break
        count = 0.0
        for j in range(k):
            if (compare[j] <= compare[0]):
                count = count + 1

        performance = performance + float(1 / count)

    return float(performance / top.__len__())

def baseline():

    valid = valid_model(valid_set_size, 10)
    test = test_model(test_set_size, 10)
    print valid
    print test

if __name__ == '__main__':
    baseline()
    print valid_set_size
    print test_set_size