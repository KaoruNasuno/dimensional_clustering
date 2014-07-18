# -*- coding:utf-8 -*-

__authors__ = "Kaoru Nasuno"
__copyright__ = "Copyright 2014"
__credits__ = ["Kaoru Nasuno"]
__license__ = "GPL"

import numpy as np
from numpy.linalg import norm


def _validate_vector(u, dtype=None):
    u = np.asarray(u, dtype=dtype, order='c').squeeze()
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u


def cosine(u, v):
    u = _validate_vector(u)
    v = _validate_vector(v)
    return np.dot(u, v) / (norm(u) * norm(v))


def dot(u, v):
    u = _validate_vector(u)
    v = _validate_vector(v)
    return np.dot(u, v)


def correlation(u, v):
    u = _validate_vector(u)
    v = _validate_vector(v)
    umu = u.mean()
    vmu = v.mean()
    um = u - umu
    vm = v - vmu
    return np.dot(um, vm) / (norm(um) * norm(vm))


if __name__ == '__main__':
    u = np.array([1, 2, 3])
    v = np.array([0, 1, -1])

    print dot(u, v)
    print cosine(u, v)
    print correlation(u, v)

#
