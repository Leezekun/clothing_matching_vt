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



fb = open('Avg_pos_rand_total.txt', 'a+')
fa = open('mrr_rand_total.txt', 'a+')

for K in range(10,210,5):
    mrr=0.0;
    print (str(K))
    pairs=numpy.loadtxt('pair_matri_totalK'+str(K)+'_811.txt');
    pairs.shape

    sum_pos=0;
    for i in range(len(pairs)):
        mij= numpy.random.rand( len(pairs[i])-1)

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



