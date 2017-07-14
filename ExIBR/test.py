import numpy
import random
index_shuf = range(10)
xxx = random.shuffle(index_shuf)


def Normalization(x):
    A = numpy.array(x.max(axis=0) - x.min(axis=0))
    A[numpy.where(A < 0.000000000000000001)] = 0.000000001
    return (x - x.min(axis=0)) / A


a = [[1,2,6],
     [1,5,4],
     [5,2,8],
     [2,6,7]]
b=numpy.asarray(a)
c=Normalization(b)
print(c)

