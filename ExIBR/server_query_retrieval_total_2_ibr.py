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

print ('loading test_cj....')
print   ('now():'+str( datetime.now() ))
test_ijk_cj =numpy.loadtxt(open("txtVisIBR_cj.csv","r"),delimiter=" ",skiprows=0, dtype=numpy.float)
print ('loading test_ci....')
print   ('now():'+str( datetime.now() ))
test_ijk_ck=numpy.loadtxt(open("txtVisIBR_ck.csv","r"),delimiter=" ",skiprows=0,dtype=numpy.float)
print ('loading test_ck....')
print   ('now():'+str( datetime.now() ))
test_ijk_ci =numpy.loadtxt(open("txtVisIBR_ci.csv","r"),delimiter=" ",skiprows=0,dtype=numpy.float)
print ('loading test_vj....')
print   ('now():'+str( datetime.now() ))
test_ijk_vj =numpy.loadtxt(open("txtVisIBR_vj.csv","r"),delimiter=" ",skiprows=0,dtype=numpy.float)
print ('loading test_vk....')
print   ('now():'+str( datetime.now() ))
test_ijk_vk =numpy.loadtxt(open("txtVisIBR_vk.csv","r"),delimiter=" ",skiprows=0,dtype=numpy.float)
print ('loading test_vi....')
print   ('now():'+str( datetime.now() ))
test_ijk_vi=numpy.loadtxt(open("txtVisIBR_vi.csv","r"),delimiter=" ",skiprows=0,dtype=numpy.float)

#
# test_ijk_vi=numpy.ones((6215,1111))
# test_ijk_vj=numpy.ones((6215,1111))
# test_ijk_vk=numpy.ones((6215,1111))
# test_ijk_ci=numpy.ones((6215,11))
# test_ijk_cj=numpy.ones((6215,11))
# test_ijk_ck=numpy.ones((6215,11))
# for i in range(0, 6215):
#     test_ijk_vi[i]=test_ijk_vi[i]*(i+1)
#     test_ijk_vj[i]=test_ijk_vj[i]*(i+1)
#     test_ijk_vk[i]=test_ijk_vk[i]*(i+1)
#     test_ijk_ci[i]=test_ijk_ci[i]*(i+1)
#     test_ijk_cj[i]=test_ijk_cj[i]*(i+1)
#     test_ijk_ck[i]=test_ijk_ck[i]*(i+1)


test_set_xi_v=test_ijk_vi;
test_set_xj_v=test_ijk_vj;
test_set_xk_v = test_ijk_vk;  #latent features

test_set_xi_c=test_ijk_ci;
test_set_xj_c=test_ijk_cj;
test_set_xk_c = test_ijk_ck;  #latent features

fb = open('Avg_pos_ibr_total.txt', 'a+')
fa = open('mrr_ibr_total.txt', 'a+')

print ('loading top-bottom lists....')
print   ('now():'+str( datetime.now() ))
test_ijk=numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/pkl/test_ijk_shuffled_811.txt","r"),delimiter="\t",skiprows=0, dtype=float)
train_ijk=numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/pkl/train_ijk_shuffled_811.txt","r"),delimiter="\t",skiprows=0,dtype=float)

top_bottom_list = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/unique_newdataset_top_bottom_valid_pairs.txt","r"),delimiter="\t",skiprows=0,dtype=float)
top_non = []
top_have=[]
train_tops = train_ijk[:,0].tolist()
test_tops = test_ijk[:,0].tolist()
total_tops=[]

test_jbottoms = test_ijk[:,1].tolist()
total_bottoms=test_jbottoms

total_bottom_feature_v=[]
total_bottom_feature_c=[]
for i in range (0, len(test_set_xj_v)):
    total_bottom_feature_v.append(test_set_xj_v[i])
    total_bottom_feature_c.append(test_set_xj_c[i])

test_kbottoms = test_ijk[:,2].tolist()

for i in range(len(test_kbottoms)):
    if (test_kbottoms[i] not in total_bottoms):
        total_bottoms.append(test_kbottoms[i])
        total_bottom_feature_v.append(test_set_xk_v[i])
        total_bottom_feature_c.append(test_set_xk_c[i])

for i in range(len(test_tops)):
#    print  (test_tops[i])
    if (test_tops[i] not in train_tops and test_tops[i] not in top_non):
        top_non.append(test_tops[i])
    elif (test_tops[i] in train_tops and test_tops[i] not in top_have):
        top_have.append(test_tops[i])
    total_tops.append(test_tops[i])

print (str(len(top_non))+' tops have not been appeared in training' )
print (str(len(top_have))+' tops have been appeared in training' )

KK=[10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290]




for K in range(10,210,5):
    mrr=0.0;
    print (str(K))
    pairs=numpy.loadtxt('pair_matri_totalK'+str(K)+'_811.txt');
    pairs.shape

    sum_pos=0;
    for i in range(len(pairs)):
        tid=pairs[i,0]
        top_index= test_tops.index(tid)
        top_feature_vi=test_ijk_vi[top_index,:]
        top_feature_ci=test_ijk_ci[top_index,:]
        mij=numpy.zeros((len(pairs[i])-1))
        for j in range (1, len(pairs[i])):
            bid=pairs[i,j]
            bot_index= total_bottoms.index(bid)
            bottom_feature_vj=total_bottom_feature_v[bot_index]
            bottom_feature_cj=total_bottom_feature_c[bot_index]
            mij[j-1]=numpy.dot(top_feature_vi,bottom_feature_vj)+numpy.dot(top_feature_ci,bottom_feature_cj);

    #    print mij
        index_i=np.argsort(- mij)
     #   print index_i
        hit=numpy.where(index_i==0)
      #  print hit[0][0]
        sum_pos=sum_pos+hit[0][0]+1
        mrr=mrr+1/(hit[0][0]+1)

    avg_pos=float(sum_pos)/len(pairs)

    avg_mrr=mrr/len(pairs)

    fb.write(' %f\t%f\n ' % (K,avg_pos))
    fb.flush()
    fa.write(' %f\t%f\n ' % (K,avg_mrr))
    fa.flush()





