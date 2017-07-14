import cPickle
import os
import json
import numpy
from matplotlib import pylab
from PIL import Image
import numpy as np
import scipy.io as sio
import random
from  datetime import *
import gzip
#import cPickle
import pickle
import random

ijk_train_list = numpy.loadtxt(open("/storage/songxuemeng/lizekun/ExIBR/train_ijk_811_noshuffle.txt", "r"),
                         delimiter="\t", skiprows=0)
ijk_valid_list = numpy.loadtxt(open("/storage/songxuemeng/lizekun/ExIBR/valid_ijk_811_noshuffle.txt", "r"),
                         delimiter="\t", skiprows=0)
ijk_test_list = numpy.loadtxt(open("/storage/songxuemeng/lizekun/ExIBR/test_ijk_811_noshuffle.txt", "r"),
                         delimiter="\t", skiprows=0)

train_list = ijk_train_list.tolist()
valid_list = ijk_valid_list.tolist()
test_list = ijk_test_list.tolist()

print ('check id....')
print ('now():' + str(datetime.now()))

print ('check valid....')
print ('now():' + str(datetime.now()))
for i in range(len(valid_list)):
    if(valid_list[i] in train_list):
        print 'No.%i valid is alraeady in the train dataset'% (i)

print ('check test....')
print ('now():' + str(datetime.now()))
for i in range(len(test_list)):
    if(test_list[i] in train_list):
        print 'No.%i test is alraeady in the train dataset'% (i)

print ('check test and valid....')
print ('now():' + str(datetime.now()))
for i in range(len(test_list)):
    if(test_list[i] in valid_list):
        print 'No.%i test is alraeady in the valid dataset'% (i)


'''
print ('check norm dataset....')
print ('now():' + str(datetime.now()))

read_file_train = open('AUC_new_dataset_train_811_norm.pkl', 'rb')
read_file_valid = open('AUC_new_dataset_valid_811_norm.pkl', 'rb')
read_file_test = open('AUC_new_dataset_test_811_norm.pkl', 'rb')
print 'loading train data'
train_set = numpy.asarray(pickle.load(read_file_train),dtype='float64')
print 'loading valid data'
valid_set = numpy.asarray(pickle.load(read_file_valid))
print 'loading test data'
test_set = numpy.asarray(pickle.load(read_file_test))

train_list = train_set.tolist()
valid_list = valid_set.tolist()
test_list = test_set.tolist()

print ('check valid....')
print ('now():' + str(datetime.now()))
for i in range(len(valid_list)):
    if(valid_list[i] in train_list):
        print 'No.%i valid is alraeady in the train dataset'% (i)

print ('check test....')
print ('now():' + str(datetime.now()))
for i in range(len(test_list)):
    if(test_list[i] in train_list):
        print 'No.%i test is alraeady in the train dataset'% (i)

print ('check test and valid....')
print ('now():' + str(datetime.now()))
for i in range(len(test_list)):
    if(test_list[i] in valid_list):
        print 'No.%i test is alraeady in the valid dataset'% (i)



print ('check dataset....')
print ('now():' + str(datetime.now()))

read_file_train = open('AUC_new_dataset_train_811.pkl', 'rb')
read_file_valid = open('AUC_new_dataset_valid_811.pkl', 'rb')
read_file_test = open('AUC_new_dataset_test_811.pkl', 'rb')
print 'loading train data'
train_set = numpy.asarray(pickle.load(read_file_train),dtype='float64')
print 'loading valid data'
valid_set = numpy.asarray(pickle.load(read_file_valid))
print 'loading test data'
test_set = numpy.asarray(pickle.load(read_file_test))

train_list = train_set.tolist()
valid_list = valid_set.tolist()
test_list = test_set.tolist()

print ('check valid....')
print ('now():' + str(datetime.now()))
for i in range(len(valid_list)):
    if(valid_list[i] in train_list):
        print 'No.%i valid is alraeady in the train dataset'% (i)

print ('check test....')
print ('now():' + str(datetime.now()))
for i in range(len(test_list)):
    if(test_list[i] in train_list):
        print 'No.%i test is alraeady in the train dataset'% (i)

print ('check test and valid....')
print ('now():' + str(datetime.now()))
for i in range(len(test_list)):
    if(test_list[i] in valid_list):
        print 'No.%i test is alraeady in the valid dataset'% (i)
'''


print ('check shuffled id....')
print ('now():' + str(datetime.now()))
ijk_train_list = numpy.loadtxt(open("/storage/songxuemeng/lizekun/ExIBR/train_ijk_shuffled_811.txt", "r"),
                         delimiter="\t", skiprows=0)
ijk_valid_list = numpy.loadtxt(open("/storage/songxuemeng/lizekun/ExIBR/valid_ijk_shuffled_811.txt", "r"),
                         delimiter="\t", skiprows=0)
ijk_test_list = numpy.loadtxt(open("/storage/songxuemeng/lizekun/ExIBR/test_ijk_shuffled_811.txt", "r"),
                         delimiter="\t", skiprows=0)

train_list = ijk_train_list.tolist()
valid_list = ijk_valid_list.tolist()
test_list = ijk_test_list.tolist()

print ('check valid....')
print ('now():' + str(datetime.now()))
for i in range(len(valid_list)):
    if(valid_list[i] in train_list):
        print 'No.%i valid is alraeady in the train dataset'% (i)

print ('check test....')
print ('now():' + str(datetime.now()))
for i in range(len(test_list)):
    if(test_list[i] in train_list):
        print 'No.%i test is alraeady in the train dataset'% (i)

print ('check test and valid....')
print ('now():' + str(datetime.now()))
for i in range(len(test_list)):
    if(test_list[i] in valid_list):
        print 'No.%i test is alraeady in the valid dataset'% (i)