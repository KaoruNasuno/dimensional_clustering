# -*- coding:utf-8 -*-

"""
This is an implementation of clustering algorithm introduced in 'Mining Clustering Dimension'(http://www.icml2010.org/papers/582.pdf).

"""

__authors__ = "Kaoru Nasuno"
__copyright__ = "Copyright 2014"
__credits__ = ["Kaoru Nasuno"]
__license__ = "GPL"


import numpy as np
import similarity as sim
from scipy import linalg as LA
from scipy.cluster.vq import kmeans2 as kmeans
import warnings
warnings.simplefilter("ignore")

# TODO
#
#


def get_similarity_matrix(X, metric):
    """
    """
    if metric == 'dot':
        func = sim.dot

    elif metric == 'cosine':
        func = sim.cosine

    elif metric == 'correlation':
        func = sim.correlation

    d = len(X)
    matrix = np.zeros((d, d))

    for i in range(d):
        for h in range(i + 1, d):
            s = func(X[i], X[h])
            matrix[i][h] = s
            matrix[h][i] = s

    return matrix


def get_diagonal_matrix(sim_matrix):
    d = len(sim_matrix)
    matrix = np.zeros((d, d))
    for i in range(d):
        matrix[i][i] = sim_matrix[i].sum()

    return matrix


def get_Laplacian_matrix(sim_matrix, diagonal_matrix):
    D_sqrtm_inverse = LA.inv(LA.sqrtm(diagonal_matrix))
    Laplacian_matrix = D_sqrtm_inverse.dot(D_sqrtm_inverse - sim_matrix).dot(D_sqrtm_inverse)
    return Laplacian_matrix


def get_partitions(Laplacian_matrix, max_dimension=4, k=2):
    w, v = LA.eig(Laplacian_matrix)

    partitions = []
    for i in np.argsort(w)[1:]:
        #print w[i], v[:, i]
        f = v[:, i]
        f = f.reshape((len(f), 1))
        centroid, partition = kmeans(f, k)
        partitions.append(partition)

        if len(partitions) >= max_dimension:
            break

    return partitions


def dimensional_clustering(X, metric, max_dimension=4, k=2):
    sim_matrix = get_similarity_matrix(X, metric)

    diagonal_matrix = get_diagonal_matrix(sim_matrix)

    Laplacian_matrix = get_Laplacian_matrix(sim_matrix, diagonal_matrix)
    partitions = get_partitions(Laplacian_matrix, max_dimension=max_dimension, k=k)

    return partitions


if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.metrics import accuracy_score

    def test():
        X = [
            [1, 2, 3, 4, 5],
            [0, 1, -1, 3, -1],
            [3, 1, 1, 3, -1]
        ]
        X = np.array(X)
        metric = 'cosine'
        metric = 'dot'
        print metric

    def test_iris():
        iris = datasets.load_iris()
        X = iris.data
        Y = iris.target

        metric = 'cosine'
        partitions = dimensional_clustering(X, metric, k=2)
        for partition in partitions:
            print partition
            print accuracy_score(partition, Y)

    test_iris()

#
