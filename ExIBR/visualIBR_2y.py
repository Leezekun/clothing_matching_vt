import os
import sys
import time
import numpy
import theano
import theano.tensor as T
import datetime
from theano.tensor.shared_randomstreams import RandomStreams
import gzip
# import cPickle
import pickle
import random


class IBR(object):
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            input1=None,
            input2=None,
            input3=None,
            n_feature=4096,
            n_y=None,
            Y1=None,
            Y2=None,
            momentum=0.8,  # 0.9
            c=None
    ):
        self.xi = input1
        self.xj = input2
        self.xk = input3
        self.momentum = momentum
        self.c = c
        self.y = n_y

        if not Y1:
            initial_Y1 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_feature + self.y)),
                    high=4 * numpy.sqrt(6. / (n_feature + self.y)),
                    size=(n_feature, self.y)
                ),
                dtype=theano.config.floatX
            )
            Y1 = theano.shared(value=initial_Y1, name='Y1', borrow=True)

        if not Y2:
            initial_Y2 = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_feature + self.y)),
                    high=4 * numpy.sqrt(6. / (n_feature + self.y)),
                    size=(n_feature, self.y)
                ),
                dtype=theano.config.floatX
            )
            Y2 = theano.shared(value=initial_Y2, name='Y2', borrow=True)

        self.Y1 = Y1
        self.Y2 = Y2
        self.params = [self.Y1, self.Y2]

    def get_cost_updates(self, learning_rate):
        y_cost = T.mean(self.Y1 ** 2) + T.mean(self.Y2 ** 2)
        '''
        s_ij = T.mean((T.dot((self.xi - self.xj), self.Y) ** 2), 1)
        s_ik = T.mean((T.dot((self.xi - self.xk), self.Y) ** 2), 1)
        '''
        s_ij = T.mean((T.dot(self.xi, self.Y1) - T.dot(self.xj, self.Y2)), 1)
        s_ik = T.mean((T.dot(self.xi, self.Y1) - T.dot(self.xk, self.Y2)), 1)
        #        theta_ij = T.log(T.nnet.sigmoid(-s_ij + self.c))
        #        theta_ik = T.log(1 - T.nnet.sigmoid(-s_ik + self.c))
        theta_ij = T.tanh(-s_ij + self.c)
        theta_ik = T.tanh(-s_ik + self.c)
        d_ij = T.mean(theta_ij)
        d_ik = T.mean(theta_ik)
        #        d_ik = T.mean(T.log(1 - theta_ik))
        #        cost = -(d_ij + d_ik) + y_cost
        cost = -(d_ij - d_ik)

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

        return (cost, updates, T.mean(s_ij), T.mean(s_ik))


def test_dA(epoch_time):
    fb = open('best_ibr.txt', 'a+')
    fi = open('info_ibr.txt', 'a+')

    print 'loading data'

    read_file_train = open('AUC_new_dataset_train_811_norm.pkl', 'rb')
    read_file_valid = open('AUC_new_dataset_valid_811_norm.pkl', 'rb')
    read_file_test = open('AUC_new_dataset_test_811_norm.pkl', 'rb')

    train_set = numpy.asarray(pickle.load(read_file_train))
    valid_set = numpy.asarray(pickle.load(read_file_valid))
    test_set = numpy.asarray(pickle.load(read_file_test))

    train_set_size = train_set[0].shape[0]
    valid_set_size = valid_set[0].shape[0]
    test_set_size = test_set[0].shape[0]

    train_set_xi, train_set_xj, train_set_xk = theano.shared(train_set[0]), theano.shared(train_set[1]), theano.shared(
        train_set[2])
    valid_set_xi, valid_set_xj, valid_set_xk = valid_set[0], valid_set[1], valid_set[2]
    test_set_xi, test_set_xj, test_set_xk = test_set[0], test_set[1], test_set[2]

    print 'loaded data'

    index = T.lscalar()
    xi = T.matrix('xi', dtype='float32')
    xj = T.matrix('xj', dtype='float32')
    xk = T.matrix('xk', dtype='float32')
    numpy_rng = numpy.random.RandomState(123)
    best_validation_score = 0.0
    best_iter = [0.0, 0.0, 0.0]
    test_score = 0.0
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    for batch_size in [128, 64, 200, 50]:
        for learning_rate in [0.5, 0.1, 1]:
            for y in [10, 20, 50, 100]:

                n_train_batches = int(train_set_size / batch_size) + 1
                print n_train_batches

                iter = [learning_rate, batch_size, y]

                ibr = IBR(
                    numpy_rng=numpy_rng,
                    input1=xi,
                    input2=xj,
                    input3=xk,
                    n_y=y,
                    c=0
                )

                ###############
                # TRAIN MODEL #
                ###############

                cost, updates, s_ij, s_ik = ibr.get_cost_updates(
                    learning_rate=learning_rate)

                train_model = theano.function(
                    [index],
                    [cost, s_ij, s_ik],
                    updates=updates,
                    givens={
                        xi: train_set_xi[index * batch_size: (index + 1) * batch_size],
                        xj: train_set_xj[index * batch_size: (index + 1) * batch_size],
                        xk: train_set_xk[index * batch_size: (index + 1) * batch_size],
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

                    print 'learning_rate is %f, batch_size is %f, y is %f, %i epochs train ended, cost is %f, s_ij is %f, s_ik is %f' % (
                    learning_rate, batch_size, y, epoch, cost, s_ij, s_ik)

                    fi.write(
                        'learning_rate is %f, batch_size is %f, y is %f, %i epochs train ended, cost is %f, s_ij is %f, s_ik is %f' % (
                        learning_rate, batch_size, y, epoch, cost, s_ij, s_ik))
                    fi.flush()

                    print 'train ended'

                    def valid_model(size, k):
                        vi = valid_set_xi
                        vj = valid_set_xj
                        vk = valid_set_xk
                        vk_list = vk.tolist()
                        y = ibr.Y.get_value()
                        ij = numpy.sum((numpy.dot((vi - vj), y) ** 2), 1)
                        ik = numpy.sum((numpy.dot((vi - vk), y) ** 2), 1)
                        top = []
                        performance = 0.0
                        vi_list = vi.tolist()
                        for i in range(size):
                            if (vi_list[i] not in top):
                                top.append(vi_list[i])
                        for i in range(top.__len__()):
                            t = top[i]
                            compare = []
                            positive = []
                            first_j = 0
                            for j in range(size):
                                if (vi_list[j] == t):
                                    positive.append(vj[j].tolist())
                                    first_j = j
                            compare.append(ij[first_j])
                            k_list = random.sample(range(size), 30)
                            for j in k_list:
                                if (vk_list[j] not in positive):
                                    mik = numpy.sum((numpy.dot((numpy.asarray(t) - numpy.asarray(vk[j])), y) ** 2))
                                    compare.append(mik)
                                if (compare.__len__() == k):
                                    break
                            count = 0.0
                            for j in range(k):
                                if (compare[j] <= compare[0]):
                                    count = count + 1

                            performance = performance + float(1 / count)

                        return float(performance / top.__len__())

                    def test_model(size, k):
                        ti = test_set_xi
                        tj = test_set_xj
                        tk = test_set_xk
                        tk_list = tk.tolist()
                        y = ibr.Y.get_value()
                        ij = numpy.sum((numpy.dot((ti - tj), y) ** 2), 1)
                        ik = numpy.sum((numpy.dot((ti - tk), y) ** 2), 1)
                        top = []
                        performance = 0.0
                        ti_list = ti.tolist()
                        for i in range(size):
                            if (ti_list[i] not in top):
                                top.append(ti_list[i])
                        for i in range(top.__len__()):
                            t = top[i]
                            compare = []
                            positive = []
                            first_j = 0
                            for j in range(size):
                                if (ti_list[j] == t):
                                    positive.append(tj[j].tolist())
                                    first_j = j
                            compare.append(ij[first_j])
                            k_list = random.sample(range(size), 30)
                            for j in k_list:
                                if (tk_list[j] not in positive):
                                    mik = numpy.sum((numpy.dot((numpy.asarray(t) - numpy.asarray(tk[j])), y) ** 2))
                                    compare.append(mik)
                                if (compare.__len__() == k):
                                    break
                            count = 0.0
                            for j in range(k):
                                if (compare[j] <= compare[0]):
                                    count = count + 1

                            performance = performance + float(1 / count)

                        return float(performance / top.__len__())

                    def valid_model2(size, k):

                        valid_data = numpy.loadtxt('valid_k_10.csv')
                        vi = valid_set_xi
                        vj = valid_set_xj
                        vk = valid_set_xk
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
                                (numpy.dot(numpy.asarray(vi[xi]), y1) - numpy.dot(numpy.asarray(vj[xj]), y2)) ** 2)
                            compare.append(vij)
                            for j in range(k - 1):
                                xk = int(valid_data[i][j + 2])
                                vik = numpy.sum(
                                    (numpy.dot(numpy.asarray(vi[xi]), y1) - numpy.dot(numpy.asarray(vk[xk]), y2)) ** 2)
                                compare.append(vik)

                            for j in range(k):
                                if (compare[j] <= compare[0]):
                                    count = count + 1
                            performance = performance + float(1 / count)

                        return float(performance / valid_data.shape[0])

                    def test_model2(size, k):

                        test_data = numpy.loadtxt('test_k_10.csv')
                        ti = numpy.asarray(test_set[0])
                        tj = numpy.asarray(test_set[1])
                        tk = numpy.asarray(test_set[2])
                        y1 = ibr.Y1.get_value()
                        y2 = ibr.Y2.get_value()
                        top = []
                        performance = 0.0
                        for i in range(test_data.shape[0]):
                            count = 0.0
                            compare = []
                            xi = int(test_data[i][0])
                            xj = int(test_data[i][1])
                            tij = numpy.sum(
                                (numpy.dot(numpy.asarray(ti[xi]), y1) - numpy.dot(numpy.asarray(tj[xj]), y2)) ** 2)
                            compare.append(tij)
                            for j in range(k - 1):
                                xk = int(test_data[i][j + 2])
                                tik = numpy.sum(
                                    (numpy.dot(numpy.asarray(ti[xi]), y1) - numpy.dot(numpy.asarray(tk[xk]), y2)) ** 2)
                                compare.append(tik)

                            for j in range(k):
                                if (compare[j] <= compare[0]):
                                    count = count + 1
                            performance = performance + float(1 / count)

                        return float(performance / test_data.shape[0])

                    print 'validation start'
                    this_validation_score = valid_model2(valid_set_size, 10)
                    print 'validation ended '

                    print 'learning_rate is %f, batch_size is %f, y is %f, %i epoch, validation score is %f' % (
                    iter[0], iter[1], iter[2], epoch, this_validation_score)
                    fi.write('learning_rate is %f, batch_size is %f, y is %f, %i epoch, validation score is %f\n' % (
                    iter[0], iter[1], iter[2], epoch, this_validation_score))
                    fi.flush()

                    #########################
                    # TEST MODEL And RECORD #
                    ########################
                    if this_validation_score > best_validation_score:
                        best_validation_score = this_validation_score
                        best_iter = iter
                        numpy.savetxt('ibr_y1.csv', ibr.Y1.get_value(), fmt="%f")
                        numpy.savetxt('ibr_y2.csv', ibr.Y2.get_value(), fmt="%f")
                        print 'test start'
                        test_score = test_model2(test_set_size, 10)
                        print 'test ended'
                        print 'test_score is %f' % (test_score)

                    print '%i epoch ended, best_learning_rate is %f, best_batch_size is %f, y is %f, best validation score is %f,' \
                          'test score is %f' % (
                          epoch, best_iter[0], best_iter[1], best_iter[2], best_validation_score, test_score)

    print 'best_learning_rate is %f, best_batch_size is %f, best_y is %f, best validation score is %f,' \
          'test score is %f' % (best_iter[0], best_iter[1], best_iter[2], best_validation_score, test_score)
    end_time = time.clock()
    print 'running time is %f' % (end_time - start_time)
    fi.write('running time is %f' % (end_time - start_time))
    fi.flush()

    fi.close()
    fb.close()


if __name__ == '__main__':
    test_dA(30)