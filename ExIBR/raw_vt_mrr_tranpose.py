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
read_file_valid = open('AUC_new_dataset_unified_text_valid8110_10000.pkl', 'rb')
valid_set = numpy.asarray(pickle.load(read_file_valid))

print 'loading text test data'
print ('now():'+str( datetime.now() ))
read_file_test = open('AUC_new_dataset_unified_text_test8110_10000.pkl', 'rb')
test_set = numpy.asarray(pickle.load(read_file_test))


valid_set_xi, valid_set_xj, valid_set_xk = valid_set[0], valid_set[1], valid_set[2]
test_set_xi, test_set_xj, test_set_xk = test_set[0], test_set[1], test_set[2]

print 'loading visual valid data'
print ('now():'+str( datetime.now() ))
read_file_valid = open('AUC_new_dataset_valid_811_norm_10000.pkl', 'rb')

print 'loading visual test data'
print ('now():'+str( datetime.now() ))
read_file_test = open('AUC_new_dataset_test_811_norm_10000.pkl', 'rb')


valid_set_visual = numpy.asarray(pickle.load(read_file_valid))
test_set_visual = numpy.asarray(pickle.load(read_file_test))


valid_set_visual_xi, valid_set_visual_xj, valid_set_visual_xk = valid_set_visual[0], valid_set_visual[1], valid_set_visual[2]
test_set_visual_xi, test_set_visual_xj, test_set_visual_xk = test_set_visual[0], test_set_visual[1], test_set_visual[2]

valid_set_size = valid_set[0].shape[0]
test_set_size = test_set[0].shape[0]


def valid_model(beta, k):
    valid_data = numpy.loadtxt('valid_k_10_10000.csv')
    vi_v = numpy.asarray(valid_set_visual_xi)
    vj_v = numpy.asarray(valid_set_visual_xj)
    vj_v_t = numpy.transpose(vj_v)
    vk_v = numpy.asarray(valid_set_visual_xk)
    vk_v_t = numpy.transpose(vk_v)
    vi_c = numpy.asarray(valid_set_xi)
    vj_c = numpy.asarray(valid_set_xj)
    vj_c_t = numpy.transpose(vj_c)
    vk_c = numpy.asarray(valid_set_xk)
    vk_c_t = numpy.transpose(vk_c)
    vij_v = numpy.dot(vi_v, vj_v_t)
    vik_v = numpy.dot(vi_v, vk_v_t)
    vij_c = numpy.dot(vi_c, vj_c_t)
    vik_c = numpy.dot(vi_c, vk_c_t)
    vi_list = vi_v.tolist()
    vk_list = vk_v.tolist()
    top = []
    performance = 0.0
    for i in range(valid_data.shape[0]):
        count = 0.0
        compare = []
        xi = int(valid_data[i][0])
        xj = int(valid_data[i][1])
        vij = vij_v[xi][xj]
        cij = vij_c[xi][xj]
        ij = (1 - beta) * vij + beta * cij
        compare.append(ij)
        for j in range(k - 1):
            xk = int(valid_data[i][j + 2])
            vik = vik_v[xi][xk]
            cik = vik_c[xi][xk]
            ik = (1 - beta) * vik + beta * cik
            compare.append(ik)

        for j in range(k):
            if (compare[j] <= compare[0]):
                count = count + 1
        performance = performance + float(1 / count)

    return float(performance / valid_data.shape[0])


def test_model(beta, k):
    test_data = numpy.loadtxt('test_k_10_10000.csv')
    ti_v = numpy.asarray(test_set_visual_xi)
    tj_v = numpy.asarray(test_set_visual_xj)
    tj_v_t = numpy.transpose(tj_v)
    tk_v = numpy.asarray(test_set_visual_xk)
    tk_v_t = numpy.transpose(tk_v)
    ti_c = numpy.asarray(test_set_xi)
    tj_c = numpy.asarray(test_set_xj)
    tj_c_t = numpy.transpose(tj_c)
    tk_c = numpy.asarray(test_set_xk)
    tk_c_t = numpy.transpose(tk_c)
    tij_v = numpy.dot(ti_v, tj_v_t)
    tik_v = numpy.dot(ti_v, tk_v_t)
    tij_c = numpy.dot(ti_c, tj_c_t)
    tik_c = numpy.dot(ti_c, tk_c_t)
    vi_list = ti_v.tolist()
    vk_list = tk_v.tolist()
    top = []
    performance = 0.0
    for i in range(test_data.shape[0]):
        count = 0.0
        compare = []
        xi = int(test_data[i][0])
        xj = int(test_data[i][1])
        vij = tij_v[xi][xj]
        cij = tij_c[xi][xj]
        ij = (1 - beta) * vij + beta * cij
        compare.append(ij)
        for j in range(k - 1):
            xk = int(test_data[i][j + 2])
            vik = tik_v[xi][xk]
            cik = tik_c[xi][xk]
            ik = (1 - beta) * vik + beta * cik
            compare.append(ik)

        for j in range(k):
            if (compare[j] <= compare[0]):
                count = count + 1
        performance = performance + float(1 / count)

    return float(performance / test_data.shape[0])

def baseline():
    beta = 0.5
    valid = valid_model(beta, 10)
    test = test_model(beta, 10)
    print valid
    print test

if __name__ == '__main__':
    baseline()
    print valid_set_size
    print test_set_size