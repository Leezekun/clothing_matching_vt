import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import gzip
#import cPickle
import pickle
import datetime
import random


read_file_valid = open('AUC_new_dataset_valid_811_norm_10000.pkl', 'rb')
read_file_test = open('AUC_new_dataset_test_811_norm_10000.pkl', 'rb')


print 'loading valid data'
valid_set = numpy.asarray(pickle.load(read_file_valid))

print 'loading test data'
test_set = numpy.asarray(pickle.load(read_file_test))


valid_set_size = valid_set[0].shape[0]
test_set_size = test_set[0].shape[0]

valid_set_xi_v, valid_set_xj_v, valid_set_xk_v = theano.shared(valid_set[0]), theano.shared(valid_set[1]), theano.shared(valid_set[2])
test_set_xi_v, test_set_xj_v, test_set_xk_v = theano.shared(test_set[0]), theano.shared(test_set[1]), theano.shared(test_set[2])

print 'loaded data'


def valid_model(size, k):
    valid_id = numpy.loadtxt('valid_ijk_811_noshuffle_10000.txt', delimiter="\t")
    f1 = open('valid_k_'+str(k)+'_10000.txt','w+')
    f2 = open('valid_k_' + str(k) + '_id_10000.txt', 'w+')
    vi = numpy.asarray(valid_set[0])
    vj = numpy.asarray(valid_set[1])
    vk = numpy.asarray(valid_set[2])
    top = []
    vi_list = vi.tolist()
    vk_list = vk.tolist()
    valid_data = []
    valid_data_id = []
    for i in range(size):
        if (vi_list[i] not in top):
            top.append(vi_list[i])
    for i in range(top.__len__()):
        compare = []
        compare_id = []
        positive = []
        positive_v = []
        t_v = top[i]
        for j in range(size):
            if (vi_list[j] == t_v):
                positive.append(j)
                positive_v.append(vj[j].tolist())
        compare.append(positive[0])
        compare.append(positive[0])
        compare_id.append(valid_id[positive[0]][0])
        compare_id.append(valid_id[positive[0]][1])
        k_list = random.sample(range(size), 3 * k)
        for j in k_list:
            if (vk_list[j] not in positive_v):
                compare.append(j)
                compare_id.append(valid_id[j][2])
            if (compare.__len__() == k+1):
                break
        valid_data.append(compare)
        valid_data_id.append(compare_id)
        for i in range(compare.__len__()):
            f1.write(str(compare[i]) + "\t")
        f1.write("\n")
        for i in range(compare_id.__len__()):
            f2.write(str(compare_id[i]) + "\t")
        f2.write("\n")

    f1.close()
    f2.close()

    numpy.savetxt('valid_k_10_10000.csv', valid_data, fmt="%d")
    numpy.savetxt('valid_k_10_id_10000.csv', valid_data_id, fmt="%d")

def test_model(size, k):
    test_id = numpy.loadtxt('test_ijk_811_noshuffle_100000.txt', delimiter="\t")
    f1 = open('test_k_' + str(k) + '_10000.txt', 'w+')
    f2 = open('test_k_' + str(k) + '_id_10000.txt', 'w+')
    vi = numpy.asarray(test_set[0])
    vj = numpy.asarray(test_set[1])
    vk = numpy.asarray(test_set[2])
    top = []
    vi_list = vi.tolist()
    vk_list = vk.tolist()
    test_data = []
    test_data_id = []
    for i in range(size):
        if (vi_list[i] not in top):
            top.append(vi_list[i])
    for i in range(top.__len__()):
        compare = []
        compare_id = []
        positive = []
        positive_v = []
        t_v = top[i]
        for j in range(size):
            if (vi_list[j] == t_v):
                positive.append(j)
                positive_v.append(vj[j].tolist())
        compare.append(positive[0])
        compare.append(positive[0])
        compare_id.append(test_id[positive[0]][0])
        compare_id.append(test_id[positive[0]][1])
        k_list = random.sample(range(size), 3 * k)
        for j in k_list:
            if (vk_list[j] not in positive_v):
                compare.append(j)
                compare_id.append(test_id[j][2])
            if (compare.__len__() == k+1):
                break
        test_data.append(compare)
        test_data_id.append(compare_id)
        for i in range(compare.__len__()):
            f1.write(str(compare[i]) + "\t")
        f1.write("\n")
        for i in range(compare_id.__len__()):
            f2.write(str(compare_id[i]) + "\t")
        f2.write("\n")

    f1.close()
    f2.close()

    numpy.savetxt('test_k_10_10000.csv', test_data, fmt="%d")
    numpy.savetxt('test_k_10_id_10000.csv', test_data_id, fmt="%d")

if __name__ == '__main__':
    valid_model(valid_set_size, 10)
    test_model(test_set_size, 10)