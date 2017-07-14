# coding=utf-8
import pickle
import os
import json
import numpy
from matplotlib import pylab
from PIL import Image
import numpy as np
import scipy.io as sio
import random
from  datetime  import  *


# dataFile = 'D:/PycharmProjects/liujinhuan/fc71.mat'
# data = sio.loadmat(dataFile)

read_file_test = open('AUC_new_dataset_test_811_norm.pkl', 'rb')
read_file_txt_test = open('AUC_new_dataset_unified_text_test8110.pkl', 'rb')
print 'loading test data'
print   ('now():'+str( datetime.now() ))
test_set = numpy.asarray(pickle.load(read_file_test))
test_txt_set = numpy.asarray(pickle.load(read_file_txt_test))


test_ijk_vi, test_ijk_vj, test_ijk_vk =test_set[0], test_set[1], test_set[2]

test_ijk_ci, test_ijk_cj, test_ijk_ck = test_txt_set[0], test_txt_set[1], test_txt_set[2]


test_set_xi=test_ijk_vi;
test_set_xj=test_ijk_vj;
test_set_xk = test_ijk_vk;  #latent features


print ('loading top-bottom lists....')
print   ('now():'+str( datetime.now() ))
test_ijk=numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/pkl/test_ijk_shuffled_811.txt","r"),delimiter="\t",skiprows=0, dtype=float)
train_ijk=numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/pkl/train_ijk_shuffled_811.txt","r"),delimiter="\t",skiprows=0,dtype=float)

top_bottom_list = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/unique_newdataset_top_bottom_valid_pairs.txt","r"),delimiter="\t",skiprows=0,dtype=float)
top_non = []
top_have=[]
train_tops = train_ijk[:,0].tolist()
test_tops = test_ijk[:,0].tolist()
fb = open('Avg_pos_DICT_TOTAL.txt', 'a+')
fa = open('mrr_DICT_TOTAL.txt', 'a+')

test_jbottoms = test_ijk[:,1].tolist()
total_bottoms=test_jbottoms

total_bottom_feature=[]
for i in range (0, len(test_set_xj)):
    total_bottom_feature.append(test_set_xj[i])

test_kbottoms = test_ijk[:,2].tolist()

for i in range(len(test_kbottoms)):
    if (test_kbottoms[i] not in total_bottoms):
        total_bottoms.append(test_kbottoms[i])
        total_bottom_feature.append(test_set_xk[i])

sim_matrix=numpy.zeros((len(top_non),9)); #[similarity between each top_have and all 10 samples]
num=0;

#load the pairs.txt


for K in range(10,210,5):
    mrr=0.0;
    print (str(K))
    pairs=numpy.loadtxt('pair_matri_totalK'+str(K)+'_811.txt');
    pairs.shape

    sum_pos=0;
    for i in range(len(pairs)):
        tid=pairs[i,0]
        top_index= test_tops.index(tid)
        top_feature=test_set_xi[top_index,:]
        mij=numpy.zeros((len(pairs[i])-1))
        for j in range (1, len(pairs[i])):
            bid=pairs[i,j]
            bot_index= total_bottoms.index(bid)
            bottom_feature=total_bottom_feature[bot_index]
            mij[j-1]=numpy.dot(top_feature, bottom_feature)


        index_i=np.argsort(- mij)

        hit=numpy.where(index_i==0)

        sum_pos=sum_pos+hit[0][0]+1
        mrr=mrr+1/(hit[0][0]+1)


    avg_pos=float(sum_pos)/len(pairs)

    avg_mrr=mrr/len(pairs)
    np.savetxt('dict_sim_matrix'+str(K)+'.txt', sim_matrix, fmt='%.5f')
    fb.write(' %f\t%f\n ' % (K,avg_pos))
    fb.flush()
    fa.write(' %f\t%f\n ' % (K,avg_mrr))
    fa.flush()

