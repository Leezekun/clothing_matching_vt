import os
import sys
import time
import numpy
from rank_metrics import ndcg_at_k
import gzip
#import cPickle
import pickle
from  datetime  import  *

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

def valid_model(size, beta):
    ci = numpy.asarray(valid_set_xi)
    cj = numpy.transpose(numpy.asarray(valid_set_xj))
    ck = numpy.transpose(numpy.asarray(valid_set_xk))
    vi = numpy.asarray(valid_set_visual_xi)
    vj = numpy.transpose(numpy.asarray(valid_set_visual_xj))
    vk = numpy.transpose(numpy.asarray(valid_set_visual_xk))
    vij = numpy.dot(vi, vj)
    vik = numpy.dot(vi, vk)
    cij = numpy.dot(ci, cj)
    cik = numpy.dot(ci, ck)
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
                sup = (vij[j][j] - vik[j][j]) + beta * (cij[j][j] - cij[j][j])
                if (sup > 0):
                    count2 = count2 + 1
        performance = performance + float(count2 / count1)

    return float(performance / top.__len__())



def test_model(size, beta):
    ci = numpy.asarray(test_set_xi)
    cj = numpy.transpose(numpy.asarray(test_set_xj))
    ck = numpy.transpose(numpy.asarray(test_set_xk))
    vi = numpy.asarray(test_set_visual_xi)
    vj = numpy.transpose(numpy.asarray(test_set_visual_xj))
    vk = numpy.transpose(numpy.asarray(test_set_visual_xk))
    vij = numpy.dot(vi, vj)
    vik = numpy.dot(vi, vk)
    cij = numpy.dot(ci, cj)
    cik = numpy.dot(ci, ck)
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
                sup = (vij[j][j] - vik[j][j]) + beta * (cij[j][j] - cik[j][j])
                if (sup > 0):
                    count2 = count2 + 1
        performance = performance + float(count2 / count1)

    return float(performance / top.__len__())

def baseline():
    '''
    for _beta in range(1,30,1):
        beta=0.05+_beta*0.01
        print beta
        '''
    beta = 1
    valid = valid_model(valid_set_size,beta)
    test = test_model(test_set_size,beta)
    print valid
    print test

if __name__ == '__main__':
    baseline()
    print valid_set_size
    print test_set_size