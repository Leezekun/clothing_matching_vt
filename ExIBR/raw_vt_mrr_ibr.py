import os
import sys
import time
import numpy
import gzip
#import cPickle
import pickle
from  datetime  import  *
import random

print 'loading text valid data'
print ('now():'+str( datetime.now() ))
read_file_valid = open('AUC_new_dataset_unified_text_valid8110.pkl', 'rb')
valid_set = numpy.asarray(pickle.load(read_file_valid))

print 'loading text test data'
print ('now():'+str( datetime.now() ))
read_file_test = open('AUC_new_dataset_unified_text_test8110.pkl', 'rb')
test_set = numpy.asarray(pickle.load(read_file_test))


valid_set_xi, valid_set_xj, valid_set_xk = valid_set[0], valid_set[1], valid_set[2]
test_set_xi, test_set_xj, test_set_xk = test_set[0], test_set[1], test_set[2]

print 'loading visual valid data'
print ('now():'+str( datetime.now() ))
read_file_valid = open('AUC_new_dataset_valid_811_norm.pkl', 'rb')

print 'loading visual test data'
print ('now():'+str( datetime.now() ))
read_file_test = open('AUC_new_dataset_test_811_norm.pkl', 'rb')


valid_set_visual = numpy.asarray(pickle.load(read_file_valid))
test_set_visual = numpy.asarray(pickle.load(read_file_test))


valid_set_visual_xi, valid_set_visual_xj, valid_set_visual_xk = valid_set_visual[0], valid_set_visual[1], valid_set_visual[2]
test_set_visual_xi, test_set_visual_xj, test_set_visual_xk = test_set_visual[0], test_set_visual[1], test_set_visual[2]

valid_set_size = valid_set[0].shape[0]
test_set_size = test_set[0].shape[0]


def valid_model(size, k):
    vi = numpy.asarray(valid_set_visual_xi)
    vj = numpy.asarray(valid_set_visual_xj)
    vk = numpy.asarray(valid_set_visual_xk)
    ci = numpy.asarray(valid_set_xi)
    cj = numpy.asarray(valid_set_xj)
    ck = numpy.asarray(valid_set_xk)
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
        t_c = []
        t_v = top[i]
        for j in range(size):
            if (vi_list[j] == t_v):
                t_c = ci[j]
                positive.append(j)
                positive_v.append(vj[j].tolist())
        vij = numpy.sum((numpy.asarray(t_v) - numpy.asarray(positive_v[0])) ** 2)
        cij = numpy.sum((numpy.asarray(t_c) - numpy.asarray(cj[positive[0]])) ** 2)
        ij = vij + cij
        compare.append(ij)
        k_list = random.sample(range(size), 3*k)
        for j in k_list:
            if (vk_list[j] not in positive_v):
                vij = numpy.sum((numpy.asarray(t_v) - numpy.asarray(vk[j])) ** 2)
                cij = numpy.sum((numpy.asarray(t_c) - numpy.asarray(ck[j])) ** 2)
                ij = vij + cij
                compare.append(ij)
            if (compare.__len__() == k):
                break
        count = 0.0
        for j in range(k):
            if (compare[j] <= compare[0]):
                count = count + 1

        performance = performance + float(1 / count)

    return float(performance / top.__len__())


def test_model(size, k):
    vi = numpy.asarray(test_set_visual_xi)
    vj = numpy.asarray(test_set_visual_xj)
    vk = numpy.asarray(test_set_visual_xk)
    ci = numpy.asarray(test_set_xi)
    cj = numpy.asarray(test_set_xj)
    ck = numpy.asarray(test_set_xk)
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
        t_c = []
        t_v = top[i]
        for j in range(size):
            if (vi_list[j] == t_v):
                t_c = ci[j]
                positive.append(j)
                positive_v.append(vj[j].tolist())
        vij = numpy.sum((numpy.asarray(t_v) - numpy.asarray(positive_v[0])) ** 2)
        cij = numpy.sum((numpy.asarray(t_c) - numpy.asarray(cj[positive[0]])) ** 2)
        ij = vij + cij
        compare.append(ij)
        k_list = random.sample(range(size), 3*k)
        for j in k_list:
            if (vk_list[j] not in positive_v):
                vij = numpy.sum((numpy.asarray(t_v) - numpy.asarray(vk[j])) ** 2)
                cij = numpy.sum((numpy.asarray(t_c) - numpy.asarray(ck[j])) ** 2)
                ij = vij + cij
                compare.append(ij)
            if (compare.__len__() == k):
                break
        count = 0.0
        for j in range(k):
            if (compare[j] <= compare[0]):
                count = count + 1

        performance = performance + float(1 / count)

    return float(performance / top.__len__())

def valid_model1(size, k, beta):
    valid_data = numpy.loadtxt('valid_k_10.csv')
    vi = numpy.asarray(valid_set_visual_xi)
    vj = numpy.asarray(valid_set_visual_xj)
    vk = numpy.asarray(valid_set_visual_xk)
    ci = numpy.asarray(valid_set_xi)
    cj = numpy.asarray(valid_set_xj)
    ck = numpy.asarray(valid_set_xk)
    performance = 0.0
    for i in range(valid_data.shape[0]):
        count = 0.0
        compare = []
        xi = int(valid_data[i][0])
        xj = int(valid_data[i][1])
        vij = numpy.sum((numpy.asarray(vi[xi]) - numpy.asarray(vj[xj]) ** 2))
        cij = numpy.sum((numpy.asarray(ci[xi]) - numpy.asarray(cj[xj]) ** 2))
        ij = (1 - beta) * vij + beta * cij
        compare.append(ij)
        for j in range(k - 1):
            xk = int(valid_data[i][j + 2])
            vij = numpy.sum((numpy.asarray(vi[xi]) - numpy.asarray(vk[xk]) ** 2))
            cij = numpy.sum((numpy.asarray(ci[xi]) - numpy.asarray(ck[xk]) ** 2))
            ij = (1 - beta) * vij + beta * cij
            compare.append(ij)
        for j in range(k):
            if (compare[j] <= compare[0]):
                count = count + 1
        performance = performance + float(1 / count)


    return float(performance / valid_data.shape[0])

def test_model1(size, k, beta):
    test_data = numpy.loadtxt('test_k_10.csv')
    vi = numpy.asarray(test_set_visual_xi)
    vj = numpy.asarray(test_set_visual_xj)
    vk = numpy.asarray(test_set_visual_xk)
    ci = numpy.asarray(test_set_xi)
    cj = numpy.asarray(test_set_xj)
    ck = numpy.asarray(test_set_xk)
    performance = 0.0
    for i in range(test_data.shape[0]):
        count = 0.0
        compare = []
        xi = int(test_data[i][0])
        xj = int(test_data[i][1])
        vij = numpy.sum((numpy.asarray(vi[xi]) - numpy.asarray(vj[xj]) ** 2))
        cij = numpy.sum((numpy.asarray(ci[xi]) - numpy.asarray(cj[xj]) ** 2))
        ij = (1 - beta) * vij + beta * cij
        compare.append(ij)
        for j in range(k - 1):
            xk = int(test_data[i][j + 2])
            vij = numpy.sum((numpy.asarray(vi[xi]) - numpy.asarray(vk[xk]) ** 2))
            cij = numpy.sum((numpy.asarray(ci[xi]) - numpy.asarray(ck[xk]) ** 2))
            ij = (1 - beta) * vij + beta * cij
            compare.append(ij)
        for j in range(k):
            if (compare[j] <= compare[0]):
                count = count + 1
        performance = performance + float(1 / count)


    return float(performance / test_data.shape[0])

def baseline():

    valid = valid_model1(valid_set_size, 10, 0.7)
    test = test_model1(test_set_size, 10, 0.7)
    print valid
    print test

if __name__ == '__main__':
    baseline()
    print valid_set_size
    print test_set_size