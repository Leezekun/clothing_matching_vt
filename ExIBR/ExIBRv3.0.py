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

# start-snippet-1
class IBR(object):

    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            input1_v=None,
            input2_v=None,
            input3_v=None,
            input1_c=None,
            input2_c=None,
            input3_c=None,
            n_feature1=4096,
            n_feature2=3529,
            n_y1=None,
            n_y2=None,
            Y1=None,
            Y2=None,
            momentum=0.9,
            c=None,
            beta=None,
            alpha=None
    ):
        self.beta = beta
        self.alpha=alpha
        self.c = c
        self.momentum = momentum
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input1_v is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x1_v = T.dmatrix(name='input1_v')
            self.x2_v = T.dmatrix(name='input2_v')
            self.x3_v = T.dmatrix(name='input3_v')
            self.x1_c = T.dmatrix(name='input1_c')
            self.x2_c = T.dmatrix(name='input2_c')
            self.x3_c = T.dmatrix(name='input3_c')

        else:
            self.x1_v = input1_v
            self.x2_v = input2_v
            self.x3_v = input3_v
            self.x1_c = input1_c
            self.x2_c = input2_c
            self.x3_c = input3_c

        if not Y1:
            initial_Y1 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_feature1 + n_y1)),
                    high=4 * numpy.sqrt(6. / (n_feature1 + n_y1)),
                    size=(n_feature1, n_y1)
                ),
                dtype=theano.config.floatX
            )
            Y1 = theano.shared(value=initial_Y1, name='Y1', borrow=True)
        if not Y2:
            initial_Y2 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_feature2 + n_y2)),
                    high=4 * numpy.sqrt(6. / (n_feature2 + n_y2)),
                    size=(n_feature2, n_y2)
                ),
                dtype=theano.config.floatX
            )
            Y2 = theano.shared(value=initial_Y2, name='Y2', borrow=True)

        self.Y1 = Y1
        self.Y2 = Y2
        self.params = [self.Y1,self.Y2]


    def get_cost_updates(self,learning_rate):

        y_cost = T.mean(self.Y1 ** 2) + T.mean(self.Y2 ** 2)

#        L_mod = abs(T.mean(T.dot(self.x1_v, self.Y1) - T.dot(self.x1_c, self.Y2))) + abs(T.mean(T.dot(self.x2_v, self.Y1) - T.dot(self.x2_c, self.Y2))) + abs(T.mean(T.dot(self.x3_v, self.Y1) - T.dot(self.x3_c, self.Y2)))

        v_ij = T.mean(((T.dot(self.x1_v,self.Y1) - T.dot(self.x2_v, self.Y1)) ** 2), 1)
        v_ik = T.mean(((T.dot(self.x1_v, self.Y1) - T.dot(self.x3_v, self.Y1)) ** 2), 1)
        c_ij = T.mean(((T.dot(self.x1_c, self.Y2) - T.dot(self.x2_c, self.Y2)) ** 2), 1)
        c_ik = T.mean(((T.dot(self.x1_c, self.Y2) - T.dot(self.x3_c, self.Y2)) ** 2), 1)

        s_ij = ((1 - self.beta) * v_ij + self.beta * c_ij)
        s_ik = ((1 - self.beta) * v_ik + self.beta * c_ik)
        theta_ij = T.nnet.sigmoid(-s_ij)
        theta_ik = T.nnet.sigmoid(-s_ik)
        '''
        s_ij_v = T.nnet.sigmoid(-v_ij)
        s_ij_c = T.nnet.sigmoid(-c_ij)
        s_ik_v = T.nnet.sigmoid(-v_ik)
        s_ik_c = T.nnet.sigmoid(-c_ik)
        theta_ij = (1 - self.beta) * s_ij_v + self.beta * s_ij_c
        theta_ik = (1 - self.beta) * s_ik_v + self.beta * s_ik_c
        '''
        d_ij = T.mean(theta_ij)
        d_ik = T.mean(theta_ik)
        L_dis = -(d_ij - d_ik)
        cost = L_dis

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)

        # generate the list of updates
        '''
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
            ]
        '''

        updates = []
        for p, g in zip(self.params, gparams):
            mparam_i = theano.shared(numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))
            v = self.momentum * mparam_i - learning_rate * g
            updates.append((mparam_i, v))
            updates.append((p, p + v))


        return (cost, updates, T.mean(s_ij), T.mean(s_ik), L_dis)
 #       return (cost, updates, d_ij, d_ik, L_dis, L_mod)

def test_IBR(learning_rate=1, batch_size=64, epoch_time=20, max_patience=3):

    fb = open('best_ExIBR.txt', 'a+')
    fi = open('info_ExIBR.txt', 'a+')


    print 'loading data'

    read_file_train = open('AUC_new_dataset_train_811_norm_10000.pkl', 'rb')
    read_file_valid = open('AUC_new_dataset_valid_811_norm_10000.pkl', 'rb')
    read_file_test = open('AUC_new_dataset_test_811_norm_10000.pkl', 'rb')

    read_file_txt_train = open('AUC_new_dataset_unified_text_train8110_10000.pkl', 'rb')
    read_file_txt_valid = open('AUC_new_dataset_unified_text_valid8110_10000.pkl', 'rb')
    read_file_txt_test = open('AUC_new_dataset_unified_text_test8110_10000.pkl', 'rb')

    print 'loading train data'
    train_set = numpy.asarray(pickle.load(read_file_train),dtype='float64')
    train_txt_set = numpy.asarray(pickle.load(read_file_txt_train),dtype='float64')

    print 'loading valid data'
    valid_set = numpy.asarray(pickle.load(read_file_valid))
    valid_txt_set = numpy.asarray(pickle.load(read_file_txt_valid))

    print 'loading test data'
    test_set = numpy.asarray(pickle.load(read_file_test))
    test_txt_set = numpy.asarray(pickle.load(read_file_txt_test))


    train_set_size = train_set[0].shape[0]
    valid_set_size = valid_set[0].shape[0]
    test_set_size = test_set[0].shape[0]
    n_train_batches = int(train_set_size / batch_size)
    print train_set_size
    print valid_set_size
    print test_set_size

    train_set_xi_v, train_set_xj_v, train_set_xk_v = theano.shared(train_set[0]), theano.shared(train_set[1]), theano.shared(train_set[2])
    valid_set_xi_v, valid_set_xj_v, valid_set_xk_v = theano.shared(valid_set[0]), theano.shared(valid_set[1]), theano.shared(valid_set[2])
    test_set_xi_v, test_set_xj_v, test_set_xk_v = theano.shared(test_set[0]), theano.shared(test_set[1]), theano.shared(test_set[2])
    train_set_xi_c, train_set_xj_c, train_set_xk_c = theano.shared(train_txt_set[0],borrow=True), theano.shared(train_txt_set[1],borrow=True), theano.shared(train_txt_set[2],borrow=True)
    valid_set_xi_c, valid_set_xj_c, valid_set_xk_c = theano.shared(valid_txt_set[0]), theano.shared(valid_txt_set[1]), theano.shared(valid_txt_set[2])
    test_set_xi_c, test_set_xj_c, test_set_xk_c = theano.shared(test_txt_set[0]), theano.shared(test_txt_set[1]), theano.shared(test_txt_set[2])

    print 'loaded data'

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    best_validation_score = 0.0
    test_score = 0.0
    ###############
    # TRAIN MODEL #
    ###############
    print 'iteration start'
    start_time = time.clock()
    count = 0
    index = T.lscalar()
    xi_v = T.matrix('xi_v',dtype='float64')
    xj_v = T.matrix('xj_v',dtype='float64')
    xk_v = T.matrix('xk_v',dtype='float64')
    xi_c = T.matrix('xi_c',dtype='float64')
    xj_c = T.matrix('xj_c',dtype='float64')
    xk_c = T.matrix('xk_c',dtype='float64')

    # c=0.5 y=8 is the best now!!!
    # c=0.8 y=10 is the best now!!!
    # c=0 y=10 is the best now!!!

    for y1 in [250]:#
        for y2 in [20]:
            for beta in [0.0]:#0.9 is best

                n_c = 0
                iter = [y1, y2, beta]
                ibr = IBR(
                    numpy_rng=rng,
                    theano_rng=theano_rng,
                    input1_v=xi_v,
                    input2_v=xj_v,
                    input3_v=xk_v,
                    input1_c=xi_c,
                    input2_c=xj_c,
                    input3_c=xk_c,
                    n_feature1=4096,
                    n_feature2=1511,#1511,3529
                    n_y1=y1,
                    n_y2=y2,
                    c=n_c,
                    momentum=0.9,
                    beta=beta,
                    alpha=0
                )

                cost, updates, s_ij, s_ik, dis = ibr.get_cost_updates(
                    learning_rate=learning_rate)

                train_model = theano.function(
                                    [index],
                                    [cost, s_ij, s_ik, dis],
                                    updates=updates,
                                    givens={
                                        xi_v: train_set_xi_v[index * batch_size: (index + 1) * batch_size],
                                        xj_v: train_set_xj_v[index * batch_size: (index + 1) * batch_size],
                                        xk_v: train_set_xk_v[index * batch_size: (index + 1) * batch_size],
                                        xi_c: train_set_xi_c[index * batch_size: (index + 1) * batch_size],
                                        xj_c: train_set_xj_c[index * batch_size: (index + 1) * batch_size],
                                        xk_c: train_set_xk_c[index * batch_size: (index + 1) * batch_size]
                                    }
                                )

                last_train_model = theano.function(
                    [],
                    outputs=ibr.get_cost_updates(learning_rate=learning_rate)[0],
                    givens={
                        xi_v: train_set_xi_v,
                        xj_v: train_set_xj_v,
                        xk_v: train_set_xk_v,
                        xi_c: train_set_xi_c,
                        xj_c: train_set_xj_c,
                        xk_c: train_set_xk_c
                    },
                    allow_input_downcast=True,
                    name='last_train_model'
                )

                def train_model2(size, k):
                    train_mij = []
                    train_data = numpy.loadtxt('train_k_10_10000.csv')
                    vi = numpy.asarray(train_set[0])
                    vj = numpy.asarray(train_set[1])
                    vk = numpy.asarray(train_set[2])
                    ci = numpy.asarray(train_txt_set[0])
                    cj = numpy.asarray(train_txt_set[1])
                    ck = numpy.asarray(train_txt_set[2])
                    y1 = ibr.Y1.get_value()
                    y2 = ibr.Y2.get_value()
                    performance = 0.0

                    for i in range(train_data.shape[0]):
                        count = 0.0
                        compare = []
                        xi = int(train_data[i][0])
                        xj = int(train_data[i][1])
                        vij = numpy.sum(
                            (numpy.dot(numpy.asarray(vi[xi]), y1) - numpy.dot(numpy.asarray(vj[xj]), y1)) ** 2)
                        cij = numpy.sum(
                            (numpy.dot(numpy.asarray(ci[xi]), y2) - numpy.dot(numpy.asarray(cj[xj]), y2)) ** 2)
                        ij = (1 - beta) * vij + beta * cij
                        compare.append(ij)
                        for j in range(k - 1):
                            xk = int(train_data[i][j + 2])
                            vik = numpy.sum(
                                (numpy.dot(numpy.asarray(vi[xi]), y1) - numpy.dot(numpy.asarray(vk[xk]), y1)) ** 2)
                            cik = numpy.sum(
                                (numpy.dot(numpy.asarray(ci[xi]), y2) - numpy.dot(numpy.asarray(ck[xk]), y2)) ** 2)
                            ik = (1 - beta) * vik + beta * cik
                            compare.append(ik)
                        train_mij.append(compare)
                        for j in range(k):
                            if (compare[j] <= compare[0]):
                                count = count + 1
                        performance = performance + float(1 / count)
                    '''
                    numpy.savetxt('sim_train_k_' + str(k) + '_10000.csv', train_mij, fmt="%f")
                '''
                    return float(performance / train_data.shape[0])

                '''
                def train_model3(size, k):
                    vi = numpy.asarray(train_set[0])
                    vj = numpy.asarray(train_set[1])
                    vk = numpy.asarray(train_set[2])
                    ci = numpy.asarray(train_txt_set[0])
                    cj = numpy.asarray(train_txt_set[1])
                    ck = numpy.asarray(train_txt_set[2])
                    y1 = ibr.Y1.get_value()
                    y2 = ibr.Y2.get_value()
                    top = []
                    performance = 0.0
                    vi_list = vi.tolist()
                    vk_list = vk.tolist()
                    for i in range(size):
                        if (vi_list[i] not in top):
                            top.append(vi_list[i])
                    for i in range(top.__len__()):
                        compare = []
                        positive = []
                        positive_v = []
                        t_v = top[i]
                        for j in range(size):
                            if (vi_list[j] == t_v):
                                positive.append(j)
                                positive_v.append(vj[j].tolist())
                        vij = numpy.sum((numpy.dot((numpy.asarray(t_v) - numpy.asarray(positive_v[0])), y1) ** 2))
                        cij = numpy.sum((numpy.dot((numpy.asarray(ci[positive[0]]) - numpy.asarray(cj[positive[0]])), y2) ** 2))
                        ij = (1 - beta) * vij + beta * cij
                        compare.append(ij)
                        k_list = random.sample(range(size), 3 * k)
                        for j in k_list:
                            if (vk_list[j] not in positive_v):
                                vij = numpy.sum((numpy.dot((numpy.asarray(t_v) - numpy.asarray(vk[j])), y1) ** 2))
                                cij = numpy.sum((numpy.dot((numpy.asarray(ci[positive[0]]) - numpy.asarray(ck[j])), y2) ** 2))
                                ij = (1 - beta) * vij + beta * cij
                                compare.append(ij)
                            if (compare.__len__() == k):
                                break
                        count = 0.0
                        for j in range(k):
                            if (compare[j] <= compare[0]):
                                count = count + 1

                        performance = performance + float(1 / count)

                    return float(performance / top.__len__())
                    '''

                print 'train start'
                start_time = time.clock()

                last_train_cost = []
                train_score = []
                for epoch in range(epoch_time):
                    cost = 0.0
                    train_batch_score = []
                    for minibatch_index in range(n_train_batches):
                        minibatch_avg_cost = train_model(minibatch_index)
                        cost = cost + minibatch_avg_cost[0]
                        s_ij = minibatch_avg_cost[1]
                        s_ik = minibatch_avg_cost[2]
                        dis_cost = minibatch_avg_cost[3]

                        ###Train_MRR
                        '''
                        k = 10
                        vi_list = train_set[0][minibatch_index * batch_size: (minibatch_index + 1) * batch_size].tolist()
                        vj_list = train_set[1][minibatch_index * batch_size: (minibatch_index + 1) * batch_size].tolist()
                        vk_list = train_set[2][minibatch_index * batch_size: (minibatch_index + 1) * batch_size].tolist()
                        ci = train_txt_set[0][minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                        cj = train_txt_set[1][minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                        ck = train_txt_set[2][minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
                        y1 = ibr.Y1.get_value()
                        y2 = ibr.Y2.get_value()
                        top = []
                        performance = 0.0
                        for i in range(batch_size):
                            if (vi_list[i] not in top):
                                top.append(vi_list[i])
                        for i in range(top.__len__()):
                            compare = []
                            positive = []
                            positive_v = []
                            t = top[i]
                            for j in range(batch_size):
                                if (vi_list[j] == t):
                                    positive.append(j)
                                    positive_v.append(vj_list[j])
                            vij = numpy.sum(
                                (numpy.dot(numpy.asarray(t),y1) - numpy.dot(numpy.asarray(vj_list[positive[0]]), y1)) ** 2)
                            cij = numpy.sum(
                                (numpy.dot(numpy.asarray(ci[i]), y2) - numpy.dot(numpy.asarray(cj[positive[0]]),
                                                                             y2)) ** 2)
                            ij = (1 - beta) * vij + beta * cij
                            compare.append(ij)
                            k_list = random.sample(range(batch_size), 3*k)
                            for j in k_list:
                                if (vk_list[j] not in positive_v):
                                    vik = numpy.sum(
                                        (
                                        numpy.dot(numpy.asarray(t), y1) - numpy.dot(numpy.asarray(vk_list[j]),
                                                                                    y1)) ** 2)
                                    cik = numpy.sum(
                                        (numpy.dot(numpy.asarray(ci[i]), y2) - numpy.dot(numpy.asarray(ck[j]),
                                                                                         y2)) ** 2)
                                    ik = (1 - beta) * vik + beta * cik
                                    compare.append(ik)
                                if (compare.__len__() == k):
                                    break
                            count = 0.0
                            for j in range(k):
                                if (compare[j] <= compare[0]):
                                    count = count + 1

                            performance = performance + float(1 / count)

                        train_batch_score.append(performance / top.__len__())
                    '''

                    ###COVERAGE#####
                    '''
                    score = train_model2(train_set_size, 10)
                    print 'train_score is %f' % score
                    fi.write('train_score is %f' % score)
                    fi.flush()
                    train_score.append(score)

                    last_cost = last_train_model()
                    print 'last_cost is %f' % last_cost
                    fi.write('last_cost is %f' % last_cost)
                    fi.flush()
                    last_train_cost.append(last_cost)
                    '''

                    print 'y1 is %f, y2 is %f, beta is %f, %i epochs train ended, cost is %f, s_ij is %f, s_ik is %f, dis is %f' % (y1, y2, beta, epoch, cost, s_ij, s_ik, dis_cost)

                    fi.write('y1 is %f, y2 is %f, beta is %f, %i epochs train ended, cost is %f, s_ij is %f, s_ik is %f, dis is %f' % (
                    y1, y2, beta, epoch, cost, s_ij, s_ik, dis_cost))

                    fi.flush()

                    print 'train ended'

                    def valid_model(size, k):
                        vi = numpy.asarray(valid_set[0])
                        vj = numpy.asarray(valid_set[1])
                        vk = numpy.asarray(valid_set[2])
                        ci = numpy.asarray(valid_txt_set[0])
                        cj = numpy.asarray(valid_txt_set[1])
                        ck = numpy.asarray(valid_txt_set[2])
                        y1 = ibr.Y1.get_value()
                        y2 = ibr.Y2.get_value()
                        top = []
                        performance = 0.0
                        vi_list = vi.tolist()
                        vk_list = vk.tolist()
                        for i in range(size):
                            if (vi_list[i] not in top):
                                top.append(vi_list[i])
                        for i in range(top.__len__()):
                            compare = []
                            positive = []
                            positive_v = []
                            t_c = []
                            t_v = top[i]
                            for j in range(size):
                                if (vi_list[j] == t_v):
                                    t_c = ci[j]
                                    positive.append(j)
                                    positive_v.append(vj[j].tolist())
                            vij = numpy.sum((numpy.dot((numpy.asarray(t_v) - numpy.asarray(positive_v[0])), y1) ** 2))
                            cij = numpy.sum((numpy.dot((numpy.asarray(t_c) - numpy.asarray(cj[positive[0]])), y2) ** 2))
                            ij = (1 - beta) * vij + beta * cij
                            compare.append(ij)
                            k_list = random.sample(range(size), 3*k)
                            for j in k_list:
                                if (vk_list[j] not in positive_v):
                                    vij = numpy.sum((numpy.dot((numpy.asarray(t_v) - numpy.asarray(vk[j])), y1) ** 2))
                                    cij = numpy.sum((numpy.dot((numpy.asarray(t_c) - numpy.asarray(ck[j])), y2) ** 2))
                                    ij = (1 - beta) * vij + beta * cij
                                    compare.append(ij)
                                if (compare.__len__() == k):
                                    break
                            count = 0.0
                            for j in range(k):
                                if (compare[j] <= compare[0]):
                                    count = count + 1

                            performance = performance + float(1 / count)

                        return float(performance / top.__len__())

                    def test_model(size, k):
                        vi = numpy.asarray(test_set[0])
                        vj = numpy.asarray(test_set[1])
                        vk = numpy.asarray(test_set[2])
                        ci = numpy.asarray(test_txt_set[0])
                        cj = numpy.asarray(test_txt_set[1])
                        ck = numpy.asarray(test_txt_set[2])
                        y1 = ibr.Y1.get_value()
                        y2 = ibr.Y2.get_value()
                        top = []
                        performance = 0.0
                        vi_list = vi.tolist()
                        vk_list = vk.tolist()
                        test_mij = []
                        for i in range(size):
                            if (vi_list[i] not in top):
                                top.append(vi_list[i])
                        for i in range(top.__len__()):
                            compare = []
                            positive = []
                            positive_v = []
                            t_c = []
                            t_v = top[i]
                            for j in range(size):
                                if (vi_list[j] == t_v):
                                    t_c = ci[j]
                                    positive.append(j)
                                    positive_v.append(vj[j].tolist())
                            vij = numpy.sum((numpy.dot((numpy.asarray(t_v) - numpy.asarray(positive_v[0])), y1) ** 2))
                            cij = numpy.sum((numpy.dot((numpy.asarray(t_c) - numpy.asarray(cj[positive[0]])), y2) ** 2))
                            ij = (1 - beta) * vij + beta * cij
                            compare.append(ij)
                            k_list = random.sample(range(size), 3*k)
                            for j in k_list:
                                if (vk_list[j] not in positive_v):
                                    vij = numpy.sum((numpy.dot((numpy.asarray(t_v) - numpy.asarray(vk[j])), y1) ** 2))
                                    cij = numpy.sum((numpy.dot((numpy.asarray(t_c) - numpy.asarray(ck[j])), y2) ** 2))
                                    ij = (1- beta) * vij + beta * cij
                                    compare.append(ij)
                                if (compare.__len__() == k):
                                    break
                            count = 0.0

                            for j in range(k):
                                if (compare[j] <= compare[0]):
                                    count = count + 1
                            test_mij.append(compare)
                            performance = performance + float(1 / count)

                        numpy.savetxt('test_mij.csv', test_mij, fmt="%f")

                        return float(performance / top.__len__())

                    def valid_model2(size, k):
                        valid_mij = []
                        valid_data = numpy.loadtxt('valid_k_10_10000.csv')
                        vi = numpy.asarray(valid_set[0])
                        vj = numpy.asarray(valid_set[1])
                        vk = numpy.asarray(valid_set[2])
                        ci = numpy.asarray(valid_txt_set[0])
                        cj = numpy.asarray(valid_txt_set[1])
                        ck = numpy.asarray(valid_txt_set[2])
                        y1 = ibr.Y1.get_value()
                        y2 = ibr.Y2.get_value()
                        top = []
                        performance = 0.0

                        for i in range(valid_data.shape[0]):
                            count = 0.0
                            compare = []
                            xi = int(valid_data[i][0])
                            xj = int(valid_data[i][1])
                            vij = numpy.sum(
                                (numpy.dot(numpy.asarray(vi[xi]), y1) - numpy.dot(numpy.asarray(vj[xj]), y1)) ** 2)
                            cij = numpy.sum(
                                (numpy.dot(numpy.asarray(ci[xi]), y2) - numpy.dot(numpy.asarray(cj[xj]), y2)) ** 2)
                            ij = (1 - beta) * vij + beta * cij
                            compare.append(ij)
                            for j in range(k-1):
                                xk = int(valid_data[i][j+2])
                                vik = numpy.sum(
                                    (numpy.dot(numpy.asarray(vi[xi]), y1) - numpy.dot(numpy.asarray(vk[xk]), y1)) ** 2)
                                cik = numpy.sum(
                                    (numpy.dot(numpy.asarray(ci[xi]), y2) - numpy.dot(numpy.asarray(ck[xk]), y2)) ** 2)
                                ik = (1 - beta) * vik + beta * cik
                                compare.append(ik)
                            valid_mij.append(compare)
                            for j in range(k):
                                if (compare[j] <= compare[0]):
                                    count = count + 1
                            performance = performance + float(1 / count)
                        '''
                        numpy.savetxt('sim_valid_k_'+str(k)+'.csv', valid_mij, fmt="%f")
                    '''
                        return float(performance / valid_data.shape[0])

                    def test_model2(size, k):
                        test_mij = []
                        test_data = numpy.loadtxt('test_k_10_10000.csv')
                        vi = numpy.asarray(test_set[0])
                        vj = numpy.asarray(test_set[1])
                        vk = numpy.asarray(test_set[2])
                        ci = numpy.asarray(test_txt_set[0])
                        cj = numpy.asarray(test_txt_set[1])
                        ck = numpy.asarray(test_txt_set[2])
                        y1 = ibr.Y1.get_value()
                        y2 = ibr.Y2.get_value()
                        top = []
                        performance = 0.0
                        for i in range(test_data.shape[0]):
                            count = 0.0
                            compare = []
                            xi = int(test_data[i][0])
                            xj = int(test_data[i][1])
                            vij = numpy.sum(
                                (numpy.dot(numpy.asarray(vi[xi]), y1) - numpy.dot(numpy.asarray(vj[xj]), y1)) ** 2)
                            cij = numpy.sum(
                                (numpy.dot(numpy.asarray(ci[xi]), y2) - numpy.dot(numpy.asarray(cj[xj]), y2)) ** 2)
                            ij = (1 - beta) * vij + beta * cij
                            compare.append(ij)
                            for j in range(k - 1):
                                xk = int(test_data[i][j + 2])
                                vik = numpy.sum(
                                    (numpy.dot(numpy.asarray(vi[xi]), y1) - numpy.dot(numpy.asarray(vk[xk]), y1)) ** 2)
                                cik = numpy.sum(
                                    (numpy.dot(numpy.asarray(ci[xi]), y2) - numpy.dot(numpy.asarray(ck[xk]), y2)) ** 2)
                                ik = (1 - beta) * vik + beta * cik
                                compare.append(ik)
                            test_mij.append(compare)
                            for j in range(k):
                                if (compare[j] <= compare[0]):
                                    count = count + 1
                            performance = performance + float(1 / count)

                        numpy.savetxt('sim_test_k_' + str(k) + '_10000.csv', test_mij, fmt="%f")

                        return float(performance / test_data.shape[0])

                    def valid_model_auc(size, k):
                        vi = numpy.asarray(valid_set[0])
                        vj = numpy.asarray(valid_set[1])
                        vk = numpy.asarray(valid_set[2])
                        ci = numpy.asarray(valid_txt_set[0])
                        cj = numpy.asarray(valid_txt_set[1])
                        ck = numpy.asarray(valid_txt_set[2])
                        y1 = ibr.Y1.get_value()
                        y2 = ibr.Y2.get_value()
                        top = []
                        performance = 0.0
                        vi_list = vi.tolist()
                        vk_list = vk.tolist()
                        for i in range(size):
                            if(vi_list[i] not in top):
                                top.append(vi_list[i])
                        for i in range(top.__len__()):
                            t = top[i]
                            count1 = 0.0
                            count2 = 0.0
                            for j in range(size):
                                if(vi_list[j] == t):
                                    count1 = count1 + 1
                                    vij = numpy.sum(
                                        (numpy.dot(numpy.asarray(vi[j]), y1) - numpy.dot(numpy.asarray(vj[j]),
                                                                                          y1)) ** 2)
                                    cij = numpy.sum(
                                        (numpy.dot(numpy.asarray(ci[j]), y2) - numpy.dot(numpy.asarray(cj[j]),
                                                                                          y2)) ** 2)
                                    ij = (1 - beta) * vij + beta * cij
                                    vik = numpy.sum(
                                        (numpy.dot(numpy.asarray(vi[j]), y1) - numpy.dot(numpy.asarray(vk[j]),
                                                                                         y1)) ** 2)
                                    cik = numpy.sum(
                                        (numpy.dot(numpy.asarray(ci[j]), y2) - numpy.dot(numpy.asarray(ck[j]),
                                                                                         y2)) ** 2)
                                    ik = (1 - beta) * vik + beta * cik
                                    if (ij < ik):
                                        count2 = count2 + 1
                            performance = performance + float(count2 / count1)

                        return float(performance / top.__len__())

                    def test_model_auc(size, k):
                        vi = numpy.asarray(test_set[0])
                        vj = numpy.asarray(test_set[1])
                        vk = numpy.asarray(test_set[2])
                        ci = numpy.asarray(test_txt_set[0])
                        cj = numpy.asarray(test_txt_set[1])
                        ck = numpy.asarray(test_txt_set[2])
                        y1 = ibr.Y1.get_value()
                        y2 = ibr.Y2.get_value()
                        top = []
                        performance = 0.0
                        vi_list = vi.tolist()
                        vk_list = vk.tolist()
                        for i in range(size):
                            if(vi_list[i] not in top):
                                top.append(vi_list[i])
                        for i in range(top.__len__()):
                            t = top[i]
                            count1 = 0.0
                            count2 = 0.0
                            for j in range(size):
                                if(vi_list[j] == t):
                                    count1 = count1 + 1
                                    vij = numpy.sum(
                                        (numpy.dot(numpy.asarray(vi[j]), y1) - numpy.dot(numpy.asarray(vj[j]),
                                                                                          y1)) ** 2)
                                    cij = numpy.sum(
                                        (numpy.dot(numpy.asarray(ci[j]), y2) - numpy.dot(numpy.asarray(cj[j]),
                                                                                          y2)) ** 2)
                                    ij = (1 - beta) * vij + beta * cij
                                    vik = numpy.sum(
                                        (numpy.dot(numpy.asarray(vi[j]), y1) - numpy.dot(numpy.asarray(vk[j]),
                                                                                         y1)) ** 2)
                                    cik = numpy.sum(
                                        (numpy.dot(numpy.asarray(ci[j]), y2) - numpy.dot(numpy.asarray(ck[j]),
                                                                                         y2)) ** 2)
                                    ik = (1 - beta) * vik + beta * cik
                                    if (ij < ik):
                                        count2 = count2 + 1
                            performance = performance + float(count2 / count1)

                        return float(performance / top.__len__())


                    print 'validation start'
                    this_validation_score = valid_model2(valid_set_size, 10)
                    print 'validation ended '

                    print 'y1 is %f, y2 is %f, beta is %f, %i epoch, validation score is %f' % (iter[0], iter[1], iter[2], epoch,
                                                                              this_validation_score
                                                                              )
                    fi.write('y1 is %f, y2 is %f, beta is %f, %i epoch, validation score is %f\n' % (iter[0], iter[1], iter[2], epoch,
                                                                                         this_validation_score
                                                                                         ))
                    fi.flush()

                    #########################
                    # TEST MODEL And RECORD #
                    ########################
                    if this_validation_score > best_validation_score:
                        best_validation_score = this_validation_score
                        best_iter = iter

                        '''
                        numpy.savetxt('ibr_y1.csv', ibr.Y1.get_value(), fmt="%f")
                        numpy.savetxt('ibr_y2.csv', ibr.Y2.get_value(), fmt="%f")
                    '''

                        print 'test start'
                        test_score = test_model2(test_set_size, 10)
                        print 'test ended'

                        print 'test_score is %f' % (test_score)

                    print '%i epoch ended, best_y1 is %f, best_y2 is %f, best_beta is %f, best validation score is %f,' \
                    'test score is %f' % (epoch, best_iter[0], best_iter[1], best_iter[2], best_validation_score, test_score)
                    fi.write( '%i epoch ended, best_y1 is %f, best_y2 is %f, best_beta is %f, best validation score is %f,' \
                    'test score is %f' % (epoch, best_iter[0], best_iter[1], best_iter[2], best_validation_score, test_score))
                    fi.flush()

                numpy.savetxt('ExIBR_train_score_10000.csv', train_score, fmt="%f")
                numpy.savetxt('ExIBR_last_cost_10000.csv', last_train_cost, fmt="%f")

    print 'best_y1 is %f, best_y2 is %f, best_beta is %f' % (best_iter[0], best_iter[1], best_iter[2])
    end_time = time.clock()
    print 'running time is %f' % (end_time - start_time)
    fi.write('running time is %f' % (end_time - start_time))
    fi.flush()

    fi.close()
    fb.close()

if __name__ == '__main__':
    test_IBR()