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
test_ijk_cj =numpy.loadtxt(open("txtVisTestLatent_cj811.csv","r"),delimiter=" ",skiprows=0, dtype=numpy.float)
print ('loading test_ci....')
print   ('now():'+str( datetime.now() ))
test_ijk_ck=numpy.loadtxt(open("txtVisTestLatent_ck811.csv","r"),delimiter=" ",skiprows=0,dtype=numpy.float)
print ('loading test_ck....')
print   ('now():'+str( datetime.now() ))
test_ijk_ci =numpy.loadtxt(open("txtVisTestLatent_ci811.csv","r"),delimiter=" ",skiprows=0,dtype=numpy.float)
print ('loading test_vj....')
print   ('now():'+str( datetime.now() ))
test_ijk_vj =numpy.loadtxt(open("txtVisTestLatent_vj811.csv","r"),delimiter=" ",skiprows=0,dtype=numpy.float)
print ('loading test_vk....')
print   ('now():'+str( datetime.now() ))
test_ijk_vk =numpy.loadtxt(open("txtVisTestLatent_vk811.csv","r"),delimiter=" ",skiprows=0,dtype=numpy.float)
print ('loading test_vi....')
print   ('now():'+str( datetime.now() ))
test_ijk_vi=numpy.loadtxt(open("txtVisTestLatent_vi811.csv","r"),delimiter=" ",skiprows=0,dtype=numpy.float)

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

fb = open('Avg_pos_total_ourmodel.txt', 'a+')
fa = open('mrr_total_ourmodel.txt', 'a+')

print ('loading top-bottom lists....')
print   ('now():'+str( datetime.now() ))
test_ijk=numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/pkl/test_ijk_shuffled_811.txt","r"),delimiter="\t",skiprows=0, dtype=float)
train_ijk=numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/pkl/train_ijk_shuffled_811.txt","r"),delimiter="\t",skiprows=0,dtype=float)

top_bottom_list = numpy.loadtxt(open("/storage/songxuemeng/dataset_backup/filter_dataset/unique_newdataset_top_bottom_valid_pairs.txt","r"),delimiter="\t",skiprows=0,dtype=float)
top_non = []
top_have=[]
train_tops = train_ijk[:,0].tolist()
test_tops = test_ijk[:,0].tolist()


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
total_tops=[]

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




for K in range(11,202,5):
    mrr=0.0;
    sim_matrix=numpy.zeros((len(total_tops),K)); #[similarity between each top_have and all 10 samples]
    pair_matrix=numpy.zeros((len(total_tops),K)); #[bids between each top_have and all 10 samples]
    num=0;
    for i in range(0, len(total_tops)):
    #for i in range(0,1000):
        if i%1000==0:
            print (str(K)+'the '+str(i)+' top')
            print   ('now():'+str( datetime.now() ))

        tid=total_tops[i]
        top_index= test_tops.index(tid)
    #    print top_index
        if top_index==0:
            print str(tid)+" not found in unique pairs"
      #      print top_index
        bid=test_ijk[top_index,1]
        bot_index= total_bottoms.index(bid)
     #   print bot_index

        top_feature_v=test_set_xi_v[top_index,:]
        top_feature_c=test_set_xi_c[top_index,:]
        bottom_feature_v=total_bottom_feature_v[bot_index]
        bottom_feature_c=total_bottom_feature_c[bot_index]

        r1_list=random.sample(total_bottoms,3*K);
        pair_matrix[num][1]=bid
        pair_matrix[num][0]=tid
        neg_num=2;

        sim_matrix[num][1]=numpy.dot(top_feature_v,bottom_feature_v)+numpy.dot(top_feature_c,bottom_feature_c);
        sim_matrix[num][0]=0;

        for r1 in r1_list:
            if neg_num<K:
                 check_pair= numpy.where(total_bottoms==r1)[0]
                 inde= numpy.where(top_bottom_list[:,1]==r1)
                 inde1=numpy.where(top_bottom_list[inde[0],0]==tid)
                 if len(inde1[0])>0:
                       print 'the pair'+str(tid)+",  "+str(r1)+" has been paired"

                 else:
                 #    print 'not find the pair'+str(tid)+",  "+str(r1)
                     if  r1<> bid:
                  #       print 'get the '+str(neg_num)+' neg,,,,,'
                         neg_index= total_bottoms.index(r1)

                         # if neg_index[0]>=1000:
                         #     neg_index=999;

                    #     print top_feature

                         neg_feature_v=total_bottom_feature_v[neg_index]
                         neg_feature_c=total_bottom_feature_c[neg_index]
                   #      print neg_feature
                         pair_matrix[num][neg_num]=r1;
                         sim_matrix[num][neg_num]=numpy.dot(top_feature_v,neg_feature_v)+numpy.dot(top_feature_c,neg_feature_c);
                 #        print numpy.dot(top_feature,neg_feature)

                         neg_num=neg_num+1;
        num=num+1

    np.savetxt('sim_matrix_totalK'+str(K-1)+'_811.txt', sim_matrix, fmt='%.5f')
    np.savetxt('pair_matri_totalK'+str(K-1)+'_811.txt', pair_matrix,fmt='%.0f')




    pairs=numpy.loadtxt('sim_matrix_totalK'+str(K-1)+'_811.txt');
    pairs.shape

    sum_pos=0;
    for i in range(len(pairs)):

        index_i=np.argsort(- pairs[i,1:len(pairs[0])])

        hit=numpy.where(index_i==0)

        sum_pos=sum_pos+hit[0][0]+1
        mrr=mrr+1/(hit[0][0]+1)


    avg_pos=float(sum_pos)/len(pairs)
    avg_mrr=float(mrr)/len(pairs)

    fb.write(' %f\t%f\n ' % (K,avg_pos))
    fb.flush()
    fa.write(' %f\t%f\n ' % (K,avg_mrr))
    fa.flush()

