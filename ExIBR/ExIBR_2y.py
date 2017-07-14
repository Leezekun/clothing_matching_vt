import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import gzip
# import cPickle
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
            n_y=None,
            Y1_v=None,
            Y2_v=None,
            Y1_c=None,
            Y2_c=None,
            momentum=0.9,
            c=None,
            beta=None,
            alpha=None
    ):
        self.beta = beta
        self.alpha = alpha
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

        if not Y1_v:
            initial_Y1_v = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_feature1 + n_y)),
                    high=4 * numpy.sqrt(6. / (n_feature1 + n_y)),
                    size=(n_feature1, n_y)
                ),
                dtype=theano.config.floatX
            )
            Y1_v = theano.shared(value=initial_Y1_v, name='Y1_v', borrow=True)
        if not Y1_c:
            initial_Y1_c = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_feature2 + n_y)),
                    high=4 * numpy.sqrt(6. / (n_feature2 + n_y)),
                    size=(n_feature2, n_y)
                ),
                dtype=theano.config.floatX
            )
            Y1_c = theano.shared(value=initial_Y1_c, name='Y1_c', borrow=True)

        if not Y2_v:
            initial_Y2_v = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_feature1 + n_y)),
                    high=4 * numpy.sqrt(6. / (n_feature1 + n_y)),
                    size=(n_feature1, n_y)
                ),
                dtype=theano.config.floatX
            )
            Y2_v = theano.shared(value=initial_Y2_v, name='Y2_v', borrow=True)
        if not Y2_c:
            initial_Y2_c = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_feature2 + n_y)),
                    high=4 * numpy.sqrt(6. / (n_feature2 + n_y)),
                    size=(n_feature2, n_y)
                ),
                dtype=theano.config.floatX
            )
            Y2_c = theano.shared(value=initial_Y2_c, name='Y2_c', borrow=True)

        self.Y1_v = Y1_v
        self.Y1_c = Y1_c
        self.Y2_v = Y2_v
        self.Y2_c = Y2_c
        self.params = [self.Y1_v, self.Y1_c, self.Y2_v, self.Y2_c]

    def get_cost_updates(self, learning_rate):

        y_cost = T.mean(self.Y1_v ** 2) + T.mean(self.Y1_c ** 2) + T.mean(self.Y2_v ** 2) + T.mean(self.Y2_c ** 2)
        L_mod = abs(T.mean(T.dot(self.x1_v, self.Y1_v) - T.dot(self.x1_c, self.Y1_c))) + abs(
            T.mean(T.dot(self.x2_v, self.Y2_v) - T.dot(self.x2_c, self.Y2_c))) + abs(
            T.mean(T.dot(self.x3_v, self.Y2_v) - T.dot(self.x3_c, self.Y2_c)))

        v_ij = T.mean((T.dot(self.x1_v, self.Y1_v) - T.dot(self.x2_v, self.Y2_v)) ** 2, 1)
        v_ik = T.mean((T.dot(self.x1_v, self.Y1_v) - T.dot(self.x3_v, self.Y2_v)) ** 2, 1)
        c_ij = T.mean((T.dot(self.x1_c, self.Y1_c) - T.dot(self.x2_c, self.Y2_c)) ** 2, 1)
        c_ik = T.mean((T.dot(self.x1_c, self.Y1_c) - T.dot(self.x3_c, self.Y2_c)) ** 2, 1)
        s_ij = ((1 - self.beta) * v_ij + self.beta * c_ij)
        s_ik = ((1 - self.beta) * v_ik + self.beta * c_ik)
        theta_ij = T.nnet.sigmoid(-s_ij + self.c)
        theta_ik = T.nnet.sigmoid(-s_ik + self.c)
        d_ij = T.mean(theta_ij)
        d_ik = T.mean(theta_ik)
        L_dis = -(d_ij - d_ik)
        cost = L_dis + self.alpha * L_mod

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

        return (cost, updates, T.mean(s_ij), T.mean(s_ik), L_dis, L_mod)


def test_IBR(learning_rate=1, batch_size=100, epoch_time=30, max_patience=3):
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
    train_set = numpy.asarray(pickle.load(read_file_train), dtype='float64')
    train_txt_set = numpy.asarray(pickle.load(read_file_txt_train), dtype='float64')

    print 'loading valid data'
    valid_set = numpy.asarray(pickle.load(read_file_valid))
    valid_txt_set = numpy.asarray(pickle.load(read_file_txt_valid))

    print 'loading test data'
    test_set = numpy.asarray(pickle.load(read_file_test))
    test_txt_set = numpy.asarray(pickle.load(read_file_txt_test))

    train_set_size = train_set[0].shape[0]
    valid_set_size = valid_set[0].shape[0]
    test_set_size = test_set[0].shape[0]
    n_train_batches = int(train_set_size / batch_size) + 1

    train_set_xi_v, train_set_xj_v, train_set_xk_v = theano.shared(train_set[0]), theano.shared(
        train_set[1]), theano.shared(train_set[2])
    valid_set_xi_v, valid_set_xj_v, valid_set_xk_v = theano.shared(valid_set[0]), theano.shared(
        valid_set[1]), theano.shared(valid_set[2])
    test_set_xi_v, test_set_xj_v, test_set_xk_v = theano.shared(test_set[0]), theano.shared(test_set[1]), theano.shared(
        test_set[2])
    train_set_xi_c, train_set_xj_c, train_set_xk_c = theano.shared(train_txt_set[0], borrow=True), theano.shared(
        train_txt_set[1], borrow=True), theano.shared(train_txt_set[2], borrow=True)
    valid_set_xi_c, valid_set_xj_c, valid_set_xk_c = theano.shared(valid_txt_set[0]), theano.shared(
        valid_txt_set[1]), theano.shared(valid_txt_set[2])
    test_set_xi_c, test_set_xj_c, test_set_xk_c = theano.shared(test_txt_set[0]), theano.shared(
        test_txt_set[1]), theano.shared(test_txt_set[2])

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
    xi_v = T.matrix('xi_v', dtype='float64')
    xj_v = T.matrix('xj_v', dtype='float64')
    xk_v = T.matrix('xk_v', dtype='float64')
    xi_c = T.matrix('xi_c', dtype='float64')
    xj_c = T.matrix('xj_c', dtype='float64')
    xk_c = T.matrix('xk_c', dtype='float64')

    # c=0.5 y=8 is the best now!!!
    # c=0.8 y=10 is the best now!!!
    # c=0 y=10 is the best now!!!

    for y in [20]:  # 10, 50, 100, 200
        for beta in [0.8]:  # 0.1, 0.3, 0.5, 0.7, 0.8, 0.9
            for alpha in [0.0]:  # 0.01, 0,05, 0.1, 0.2, 0.3, 0.4, 0.5

                n_y = y
                n_c = 0
                iter = [n_y, beta, alpha]
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
                    n_feature2=1511,
                    n_y=n_y,
                    c=n_c,
                    momentum=0.9,
                    beta=beta,
                    alpha=alpha
                )

                cost, updates, s_ij, s_ik, dis, mod = ibr.get_cost_updates(
                    learning_rate=learning_rate)

                train_model = theano.function(
                    [index],
                    [cost, s_ij, s_ik, dis, mod],
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

                print 'train start'
                start_time = time.clock()

                for epoch in range(epoch_time):
                    cost = 0.0
                    for minibatch_index in range(n_train_batches):
                        minibatch_avg_cost = train_model(minibatch_index)
                        cost = cost + minibatch_avg_cost[0]
                        s_ij = minibatch_avg_cost[1]
                        s_ik = minibatch_avg_cost[2]
                        mod_cost = minibatch_avg_cost[4]
                        dis_cost = minibatch_avg_cost[3]
                    print 'n_y is %f, beta is %f, alpha is %f, %i epochs train ended, cost is %f, s_ij is %f, s_ik is %f, dis is %f, mod is %f' % (
                    n_y, beta, alpha, epoch, cost, s_ij, s_ik, dis_cost, mod_cost)

                    fi.write(
                        'n_y is %f, beta is %f, alpha is %f, %i epochs train ended, cost is %f, s_ij is %f, s_ik is %f, dis is %f, mod is %f' % (
                            n_y, beta, alpha, epoch, cost, s_ij, s_ik, dis_cost, mod_cost))

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
                        #                   vij = numpy.sum((numpy.dot((vi - vj), y1) ** 2), 1)
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
                            k_list = random.sample(range(size), 3 * k)
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
                            k_list = random.sample(range(size), 3 * k)
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
                        y1_v = ibr.Y1_v.get_value()
                        y2_v = ibr.Y2_v.get_value()
                        y1_c = ibr.Y1_c.get_value()
                        y2_c = ibr.Y2_c.get_value()
                        top = []
                        performance = 0.0

                        for i in range(valid_data.shape[0]):
                            count = 0.0
                            compare = []
                            xi = int(valid_data[i][0])
                            xj = int(valid_data[i][1])
                            vij = numpy.sum((numpy.dot(numpy.asarray(vi[xi]), y1_v) - numpy.dot(numpy.asarray(vj[xj]), y2_v)) ** 2)
                            cij = numpy.sum((numpy.dot(numpy.asarray(ci[xi]), y1_c) - numpy.dot(numpy.asarray(cj[xj]), y2_c)) ** 2)
                            ij = (1 - beta) * vij + beta * cij
                            compare.append(ij)
                            for j in range(k - 1):
                                xk = int(valid_data[i][j + 2])
                                vij = numpy.sum((numpy.dot(numpy.asarray(vi[xi]), y1_v) - numpy.dot(numpy.asarray(vk[xk]), y2_v)) ** 2)
                                cij = numpy.sum((numpy.dot(numpy.asarray(ci[xi]), y1_c) - numpy.dot(numpy.asarray(ck[xk]), y2_c)) ** 2)
                                ij = (1 - beta) * vij + beta * cij
                                compare.append(ij)
                            valid_mij.append(compare)
                            for j in range(k):
                                if (compare[j] <= compare[0]):
                                    count = count + 1
                            performance = performance + float(1 / count)

                        numpy.savetxt('sim_valid_k_' + str(k) + '.csv', valid_mij, fmt="%f")

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
                        y1_v = ibr.Y1_v.get_value()
                        y2_v = ibr.Y2_v.get_value()
                        y1_c = ibr.Y1_c.get_value()
                        y2_c = ibr.Y2_c.get_value()
                        top = []
                        performance = 0.0
                        for i in range(test_data.shape[0]):
                            count = 0.0
                            compare = []
                            xi = int(test_data[i][0])
                            xj = int(test_data[i][1])
                            vij = numpy.sum((numpy.dot(numpy.asarray(vi[xi]), y1_v) - numpy.dot(numpy.asarray(vj[xj]), y2_v)) ** 2)
                            cij = numpy.sum((numpy.dot(numpy.asarray(ci[xi]), y1_c) - numpy.dot(numpy.asarray(cj[xj]), y2_c)) ** 2)
                            ij = (1 - beta) * vij + beta * cij
                            compare.append(ij)
                            for j in range(k - 1):
                                xk = int(test_data[i][j + 2])
                                vij = numpy.sum((numpy.dot(numpy.asarray(vi[xi]), y1_v) - numpy.dot(numpy.asarray(vk[xk]), y2_v)) ** 2)
                                cij = numpy.sum((numpy.dot(numpy.asarray(ci[xi]), y1_c) - numpy.dot(numpy.asarray(ck[xk]), y2_c)) ** 2)
                                ij = (1 - beta) * vij + beta * cij
                                compare.append(ij)
                            test_mij.append(compare)
                            for j in range(k):
                                if (compare[j] <= compare[0]):
                                    count = count + 1
                            performance = performance + float(1 / count)

                        numpy.savetxt('sim_test_k_' + str(k) + '.csv', test_mij, fmt="%f")

                        return float(performance / test_data.shape[0])

                    print 'validation start'
                    this_validation_score = valid_model2(valid_set_size, 10)
                    print 'validation ended '

                    print 'n_y is %f, beta is %f, alpha is %f, %i epoch, validation score is %f' % (
                    iter[0], iter[1], iter[2], epoch,
                    this_validation_score
                    )
                    fi.write('n_y is %f, beta is %f, alpha is %f, %i epoch, validation score is %f\n' % (
                    iter[0], iter[1], iter[2], epoch,
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
                        numpy.savetxt('ibr_y1_v.csv', ibr.Y1_v.get_value(), fmt="%f")
                        numpy.savetxt('ibr_y1_c.csv', ibr.Y1_c.get_value(), fmt="%f")
                        numpy.savetxt('ibr_y2_v.csv', ibr.Y2_v.get_value(), fmt="%f")
                        numpy.savetxt('ibr_y2_c.csv', ibr.Y2_c.get_value(), fmt="%f")
                    '''

                        print 'test start'
                        test_score = test_model2(test_set_size, 10)
                        print 'test ended'

                        print 'test_score is %f' % (test_score)

                    print '%i epoch ended, best_y is %f, best_beta is %f, best is %f, best validation score is %f,' \
                          'test score is %f' % (
                          epoch, best_iter[0], best_iter[1], best_iter[2], best_validation_score, test_score)

    print 'best_y is %f, best_beta is %f, best_aplha is %f' % (best_iter[0], best_iter[1], best_iter[2])
    end_time = time.clock()
    print 'running time is %f' % (end_time - start_time)
    fi.write('running time is %f' % (end_time - start_time))
    fi.flush()

    fi.close()
    fb.close()


if __name__ == '__main__':
    test_IBR()