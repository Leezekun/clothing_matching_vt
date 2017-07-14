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
from  datetime  import  *

# dataFile = 'D:/PycharmProjects/liujinhuan/fc71.mat'
# data = sio.loadmat(dataFile)


# f=sio.loadmat('D:/PycharmProjects/liujinhuan/fc71.mat')
print ('loading top features....')
print   ('now():'+str( datetime.now() ))
my_matrix_top = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/features_top/myfeatures_mean_top.csv","r"),delimiter=",",skiprows=0)

print ('loading bottom features....')
print ('now():'+str( datetime.now()))
my_matrix_bottom = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/features_bottom/myfeatures_mean_bottom.csv","r"),delimiter=",",skiprows=0)
print my_matrix_bottom.shape
print my_matrix_top.shape

print ('loading top-bottom lists....')
print   ('now():'+str( datetime.now() ))
top_list = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/top_list_id.txt","r"),delimiter="\t",skiprows=0)
bottom_list = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/bottom_list_id.txt","r"),delimiter="\t",skiprows=0)
top_bottom_list = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/unique_newdataset_top_bottom_valid_pairs.txt","r"),delimiter="\t",skiprows=0)



K=1

olivettifaces1 = numpy.empty((10000 * K,4096))
olivettifaces2 = numpy.empty((10000 * K,4096))
olivettifaces3 = numpy.empty((10000 * K,4096))

ijk=numpy.empty((10000 * K,3))


f2=open('/storage/songxuemeng/lizekun/ExIBR/total_ijk_811_noshuffle_10000.txt','w')
f3=open('/storage/songxuemeng/lizekun/ExIBR/train_ijk_811_noshuffle_10000.txt','w')
f4=open('/storage/songxuemeng/lizekun/ExIBR/valid_ijk_811_noshuffle_10000.txt','w')
f5=open('/storage/songxuemeng/lizekun/ExIBR/test_ijk_811_noshuffle_10000.txt','w')

train_end = 8000
valid_end = 9000
print 'train_end'
print train_end
print 'valid_end'
print valid_end

count = 0
for i in range (0, 10000):
    if count % 2000 == 0:
        print ('the '+ str(i) + ' top-bottom pair')

      #  print 'writing training...'+str(count)
    tid = top_bottom_list[i,0]
    bid = top_bottom_list[i,1]
    top_index = numpy.where(top_list==tid)[0]
    bottom_index = numpy.where(bottom_list==bid)[0]
    top_feature = my_matrix_top[top_index]
    bottom_feature = my_matrix_bottom[bottom_index]

    #get negative
    r1_list = random.sample(bottom_list,10)
    neg_num = 0
    for r1 in r1_list:
        if neg_num < K:

             check_pair = numpy.where(bottom_list==r1)[0]
             ind1 = numpy.where(top_bottom_list[:,1]==r1)
             ind2 = numpy.where(top_bottom_list[ind1[0],0]==tid)

             if len(ind2[0])>0:
                 print 'the pair'+str(tid)+",  "+str(r1)+" has been paired"

             else:
                 if  r1 <> bid:
        #             print 'get the '+str(neg_num)+' neg,,,,,'
                     neg_index= numpy.where(bottom_list==r1)[0]
                     neg_feature = my_matrix_bottom[neg_index]
                     olivettifaces1[count*K+neg_num] = numpy.ndarray.flatten(top_feature)
                     olivettifaces2[count*K+neg_num] = numpy.ndarray.flatten(bottom_feature)
                     olivettifaces3[count*K+neg_num] = numpy.ndarray.flatten(neg_feature)
                     ijk[count*K+neg_num,0] = tid
                     ijk[count*K+neg_num,1] = bid
                     ijk[count*K+neg_num,2] = r1
                     neg_num = neg_num + 1
                     f2.write(str(tid)+"\t"+ str(bid)+"\t"+str(r1)+"\n")


    count = count + 1

olivettifaces1_train=olivettifaces1[0:train_end*K,:]
olivettifaces2_train=olivettifaces2[0:train_end*K,:]
olivettifaces3_train=olivettifaces3[0:train_end*K,:]
ijk_train=ijk[0:train_end*K,:]
for i in range(len(ijk_train)):
    f3.write(str(ijk_train[i][0])+"\t"+str(ijk_train[i][1])+"\t"+str(ijk_train[i][2])+"\n")

print 'olivettifaces1_train length'
print len(olivettifaces1_train)


olivettifaces1_valid = olivettifaces1[train_end*K:valid_end*K,:]
olivettifaces2_valid = olivettifaces2[train_end*K:valid_end*K,:]
olivettifaces3_valid = olivettifaces3[train_end*K:valid_end*K,:]
ijk_valid = ijk[train_end * K: valid_end*K,:]
for i in range(len(ijk_valid)):
    f4.write(str(ijk_valid[i][0])+"\t"+str(ijk_valid[i][1])+"\t"+str(ijk_valid[i][2])+"\n")

print 'olivettifaces1_valid length'
print len(olivettifaces1_valid)

olivettifaces1_test = olivettifaces1[valid_end * K:len(olivettifaces1),:]
olivettifaces2_test = olivettifaces2[valid_end * K:len(olivettifaces1),:]
olivettifaces3_test = olivettifaces3[valid_end * K:len(olivettifaces1),:]
ijk_test = ijk[valid_end*K:len(olivettifaces1),:]
for i in range(len(ijk_test)):
    f5.write(str(ijk_test[i][0])+"\t"+str(ijk_test[i][1])+"\t"+str(ijk_test[i][2])+"\n")

print 'olivettifaces1_test length'
print len(olivettifaces1_test)

'''
numpy.savetxt('train_data_10000.csv', numpy.asarray(olivettifaces1_train[0:100,:]), fmt="%f")
numpy.savetxt('valid_data_10000.csv', numpy.asarray(olivettifaces1_valid[0:100,:]), fmt="%f")
numpy.savetxt('test_data_10000.csv', numpy.asarray(olivettifaces1_test[0:100,:]), fmt="%f")
'''


print 'writing training'
print   ('now():'+ str( datetime.now() ))
write_file=open('/storage/songxuemeng/lizekun/ExIBR/AUC_new_dataset_train_811_10000.pkl','wb')
# cPickle.dump将train数据放入pkl文件，其中train又包含三个矩阵，第一个为top的feature,第二个为和top相搭配的bottom feature，
# 第三个为随意搭配的bottom feature.
cPickle.dump([olivettifaces1_train,olivettifaces2_train,olivettifaces3_train],write_file)
write_file.close()
print 'writing valid'
print   ('now():'+ str( datetime.now() ))
write_file=open('/storage/songxuemeng/lizekun/ExIBR/AUC_new_dataset_valid_811_10000.pkl','wb')
# cPickle.dump将train数据放入pkl文件，其中train又包含三个矩阵，第一个为top的feature,第二个为和top相搭配的bottom feature，
# 第三个为随意搭配的bottom feature.
cPickle.dump([olivettifaces1_valid,olivettifaces2_valid,olivettifaces3_valid],write_file)
write_file.close()
print 'writing test'
print   ('now():'+ str( datetime.now() ))
write_file=open('/storage/songxuemeng/lizekun/ExIBR/AUC_new_dataset_test_811_10000.pkl','wb')
# cPickle.dump将train数据放入pkl文件，其中train又包含三个矩阵，第一个为top的feature,第二个为和top相搭配的bottom feature，
# 第三个为随意搭配的bottom feature.
cPickle.dump([olivettifaces1_test,olivettifaces2_test,olivettifaces3_test],write_file)
write_file.close()