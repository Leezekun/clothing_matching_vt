


import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import gzip
import pickle

from  datetime import *
import time
theano.config.floatX= 'float32'


def test_dA(learning_rate=0.1, batch_size=64, epoch_time=30, max_patience=3):

    print 'loading test visual data'
    print('now():' + str(datetime.now()))
    with open("AUC_new_dataset_train_811_10000.pkl", "rb") as f:
        print 'train'
        train_set = numpy.asarray(pickle.load(f),dtype='float32')

    with open("AUC_new_dataset_valid_811_10000.pkl", "rb") as f:
        print 'valid'
        valid_set = numpy.asarray(pickle.load(f),dtype='float32')

    with open("AUC_new_dataset_test_811_10000.pkl", "rb") as f:
        print 'test'
        test_set = numpy.asarray(pickle.load(f),dtype='float32')

    print 'done'
    row1=train_set[0].shape[0]
    row2=valid_set[0].shape[0]
    row3=test_set[0].shape[0]

    print 'row1'
    print row1
    print 'row2'
    print row2
    print 'row3'
    print row3


    def NormalizationCol(x):
            A = numpy.array(x.max(axis=0) - x.min(axis=0))
            A[numpy.where(A<0.000000000000000001)]=0.000000001
            return  (x - x.min(axis=0)) / A

    def NormalizationRow(y):
        x = y.transpose()
        A = numpy.array(x.max(axis=0) - x.min(axis=0))
        A[numpy.where(A < 0.000000000000000001)] = 0.000000001
        return ((x - x.min(axis=0)) / A).transpose()

    all_data_xi=numpy.vstack((train_set[0],valid_set[0]))
    all_data_xi=numpy.vstack((all_data_xi, test_set[0]))
    print 'all data xi'
    print all_data_xi.shape
    print 'processing xi'
    print('now():' + str(datetime.now()))
    new_data_xi= NormalizationRow(all_data_xi)

    all_data_xj=numpy.vstack((train_set[1],valid_set[1]))
    all_data_xj=numpy.vstack((all_data_xj, test_set[1]))
    print 'all data xj'
    print all_data_xj.shape
    print 'processing xj'
    print('now():' + str(datetime.now()))
    new_data_xj= NormalizationRow(all_data_xj)

    all_data_xk=numpy.vstack((train_set[2],valid_set[2]))
    all_data_xk=numpy.vstack((all_data_xk, test_set[2]))
    print 'all data xk'
    print all_data_xk.shape
    print 'processing xk'
    print('now():' + str(datetime.now()))
    new_data_xk= NormalizationRow(all_data_xk)

    '''
    numpy.savetxt('train_data_norm.csv', numpy.asarray(all_data_xi[0:100, :]), fmt="%f")
    numpy.savetxt('valid_data_norm.csv', numpy.asarray(all_data_xj[0:100, :]), fmt="%f")
    numpy.savetxt('test_data_norm.csv', numpy.asarray(all_data_xk[0:100, :]), fmt="%f")
    '''

    write_file=open('AUC_new_dataset_train_811_norm_10000.pkl','wb')
    pickle.dump([new_data_xi[0:row1,:],new_data_xj[0:row1,:],new_data_xk[0:row1,:]],write_file)
    write_file.close()
    print new_data_xi[0:row1,:].shape


    write_file=open('AUC_new_dataset_valid_811_norm_10000.pkl','wb')
    pickle.dump([new_data_xi[row1:row1+row2,:],new_data_xj[row1:row1+row2,:],new_data_xk[row1:row1+row2,:]],write_file)
    write_file.close()
    print new_data_xi[row1:row1+row2,:].shape

    write_file=open('AUC_new_dataset_test_811_norm_10000.pkl','wb')
    pickle.dump([new_data_xi[row1+row2:row1+row2+row3,:],new_data_xj[row1+row2:row1+row2+row3,:],new_data_xk[row1+row2:row1+row2+row3,:]],write_file)
    write_file.close()
    print new_data_xi[row1+row2:row1+row2+row3,:].shape

if __name__ == '__main__':
    test_dA()
