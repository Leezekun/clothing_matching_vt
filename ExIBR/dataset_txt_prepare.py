# coding=utf-8
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

print ('loading bottom features....')
print ('now():' + str(datetime.now()))
my_matrix_text_bottom = numpy.loadtxt(
    open("/storage/songxuemeng/dataset_backup/filter_dataset/features_text/bottom_feature.txt", "r"), delimiter=",",
    skiprows=0)
print ('bottom shape')
print my_matrix_text_bottom.shape

print ('loading top features....')
print   ('now():' + str(datetime.now()))
my_matrix_text_top = numpy.loadtxt(
    open("/storage/songxuemeng/dataset_backup/filter_dataset/features_text/top_feature.txt", "r"), delimiter=",",
    skiprows=0)

print ('top shape')
print my_matrix_text_top.shape

print ('loading ids...')
print ('now():' + str(datetime.now()))
top_id_list = numpy.loadtxt(
    open("/storage/songxuemeng/dataset_backup/filter_dataset/features_text/cate_top_id_list.txt", "r"), delimiter=" ",
    skiprows=0)
bottom_id_list = numpy.loadtxt(
    open("/storage/songxuemeng/dataset_backup/filter_dataset/features_text/cate_bottom_id_list.txt", "r"),
    delimiter=" ", skiprows=0)

print ('top_id_list.shape')
print top_id_list.shape
print ('bottom_id_list.shape')
print bottom_id_list.shape




print ('loading train ijk...')
print   ('now():' + str(datetime.now()))
ijk_list = numpy.loadtxt(open("/storage/songxuemeng/lizekun/ExIBR/train_ijk_811_noshuffle.txt", "r"),
                         delimiter="\t", skiprows=0)

olivettifaces_train1 = numpy.empty((len(ijk_list), len(my_matrix_text_top[0])))
olivettifaces_train2 = numpy.empty((len(ijk_list), len(my_matrix_text_bottom[0])))
olivettifaces_train3 = numpy.empty((len(ijk_list), len(my_matrix_text_bottom[0])))

for i in range(0, len(ijk_list)):

    if i % 50 == 0:
        print ('the ' + str(i) + ' top-bottom pair')
        print   ('now():' + str(datetime.now()))

    tid = ijk_list[i, 0]
    jid = ijk_list[i, 1]
    kid = ijk_list[i, 2]

    tid_index = numpy.where(top_id_list == tid)[0]
    jid_index = numpy.where(bottom_id_list == jid)[0]
    kid_index = numpy.where(bottom_id_list == kid)[0]

    tid_feature = my_matrix_text_top[tid_index]
    jid_feature = my_matrix_text_bottom[jid_index]
    kid_feature = my_matrix_text_bottom[kid_index]
    olivettifaces_train1[i] = numpy.ndarray.flatten(tid_feature)
    olivettifaces_train2[i] = numpy.ndarray.flatten(jid_feature)
    olivettifaces_train3[i] = numpy.ndarray.flatten(kid_feature)

write_file = open('/storage/songxuemeng/lizekun/ExIBR/AUC_new_dataset_unified_text_train8110.pkl',
                  'wb')
# cPickle.dump将train数据放入pkl文件，其中train又包含三个矩阵，第一个为top的feature,第二个为和top相搭配的bottom feature，
# 第三个为随意搭配的bottom feature.
cPickle.dump([olivettifaces_train1, olivettifaces_train2, olivettifaces_train3], write_file)
write_file.close()







print ('loading valid ijk...')
print   ('now():' + str(datetime.now()))
ijk_list = numpy.loadtxt(open("/storage/songxuemeng/lizekun/ExIBR/valid_ijk_811_noshuffle.txt", "r"),
                         delimiter="\t", skiprows=0)

olivettifaces_valid1 = numpy.empty((len(ijk_list), len(my_matrix_text_top[0])))
olivettifaces_valid2 = numpy.empty((len(ijk_list), len(my_matrix_text_bottom[0])))
olivettifaces_valid3 = numpy.empty((len(ijk_list), len(my_matrix_text_bottom[0])))

for i in range(0, len(ijk_list)):
    if i % 50 == 0:
        print ('the ' + str(i) + ' top-bottom pair')
        print   ('now():' + str(datetime.now()))

    tid = ijk_list[i, 0]
    jid = ijk_list[i, 1]
    kid = ijk_list[i, 2]

    tid_index = numpy.where(top_id_list == tid)[0]
    jid_index = numpy.where(bottom_id_list == jid)[0]
    kid_index = numpy.where(bottom_id_list == kid)[0]

    tid_feature = my_matrix_text_top[tid_index]
    jid_feature = my_matrix_text_bottom[jid_index]
    kid_feature = my_matrix_text_bottom[kid_index]
    olivettifaces_valid1[i] = numpy.ndarray.flatten(tid_feature)
    olivettifaces_valid2[i] = numpy.ndarray.flatten(jid_feature)
    olivettifaces_valid3[i] = numpy.ndarray.flatten(kid_feature)

write_file = open('/storage/songxuemeng/lizekun/ExIBR/AUC_new_dataset_unified_text_valid8110.pkl',
                  'wb')
# cPickle.dump将train数据放入pkl文件，其中train又包含三个矩阵，第一个为top的feature,第二个为和top相搭配的bottom feature，
# 第三个为随意搭配的bottom feature.
cPickle.dump([olivettifaces_valid1, olivettifaces_valid2, olivettifaces_valid3], write_file)
write_file.close()







print ('loading test ijk...')
print   ('now():' + str(datetime.now()))
ijk_list = numpy.loadtxt(open("/storage/songxuemeng/lizekun/ExIBR/test_ijk_811_noshuffle.txt", "r"),
                         delimiter="\t", skiprows=0)

olivettifaces_test1 = numpy.empty((len(ijk_list), len(my_matrix_text_top[0])))
olivettifaces_test2 = numpy.empty((len(ijk_list), len(my_matrix_text_bottom[0])))
olivettifaces_test3 = numpy.empty((len(ijk_list), len(my_matrix_text_bottom[0])))

for i in range(0, len(ijk_list)):
    if i % 50 == 0:
        print ('the ' + str(i) + ' top-bottom pair')
        print   ('now():' + str(datetime.now()))

    tid = ijk_list[i, 0]
    jid = ijk_list[i, 1]
    kid = ijk_list[i, 2]

    tid_index = numpy.where(top_id_list == tid)[0]
    jid_index = numpy.where(bottom_id_list == jid)[0]
    kid_index = numpy.where(bottom_id_list == kid)[0]

    tid_feature = my_matrix_text_top[tid_index]
    jid_feature = my_matrix_text_bottom[jid_index]
    kid_feature = my_matrix_text_bottom[kid_index]
    olivettifaces_test1[i] = numpy.ndarray.flatten(tid_feature)
    olivettifaces_test2[i] = numpy.ndarray.flatten(jid_feature)
    olivettifaces_test3[i] = numpy.ndarray.flatten(kid_feature)

write_file = open('/storage/songxuemeng/lizekun/ExIBR/AUC_new_dataset_unified_text_test8110.pkl',
                  'wb')
# cPickle.dump将train数据放入pkl文件，其中train又包含三个矩阵，第一个为top的feature,第二个为和top相搭配的bottom feature，
# 第三个为随意搭配的bottom feature.
cPickle.dump([olivettifaces_test1, olivettifaces_test2, olivettifaces_test3], write_file)
write_file.close()
