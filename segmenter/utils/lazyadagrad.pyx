#!/usr/bin/env python
# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: profile=False
#cython: initializedcheck=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = ["-std=c++11"]

import numpy as np
from numpy import empty, zeros, ones
from libc.math cimport exp, log, log1p, sqrt
from libc.float cimport DBL_MIN, DBL_MAX
from sparse cimport SparseBinaryVector

cdef inline double sign(double x) nogil:
    return 1 if x >= 0 else -1

# should import from beast
cdef class LazyRegularizedAdagrad(object):

    # cdef public double[:] w   # weight vector
    # cdef public double[:] q   # sum of squared weights
    # cdef public double eta    # learning rate (assumed constant)
    # cdef public double C      # regularization constant
    # cdef int[:] u             # time of last update
    # cdef int L                # regularizer type in {1,2}
    # cdef int d                # dimensionality
    # cdef double fudge         # adagrad fudge factor paramter
    # cdef public int step      # time step of the optimization algorithm (caller is
    #                           # responsible for incrementing)

    def __init__(self, int d, int L, double C, double eta = 0.1, double fudge = 1e-4):
        self.L = L
        self.d = d
        self.fudge = fudge
        self.u = zeros(d, dtype=np.int32)
        self.q = zeros(d, dtype=np.double) + fudge
        self.w = ones(d, dtype=np.double) * -0.1
        self.C = C
        self.eta = eta
        self.step = 1

    def reset(self):
        """ reset the AdaGrad values """
        self.u = np.zeros(self.d, dtype=np.int32)
        self.q = np.zeros(self.d, dtype=np.double) + self.fudge

    def _catchup(self, int k):
        self.catchup(k)

    def _update_active(self, int k, double g):
        self.update_active(k, g)

    def finalize(self):
        for i in range(self.d):
            self.catchup(i)
        return self.w

    cdef inline double catchup(self, int k):
        "Lazy L1/L2-regularized adagrad catchup operation."
        cdef int dt
        cdef double sq
        dt = self.step - self.u[k]
        sq = sqrt(self.q[k])
        if self.L == 2:
            # Lazy L2 regularization
            self.w[k] *= (sq / (self.eta * self.C + sq)) ** dt
        elif self.L == 1:
            # Lazy L1 regularization
            self.w[k] = sign(self.w[k]) * max(0, abs(self.w[k]) - self.eta * self.C * dt / sq)
        # update last seen
        self.u[k] = self.step
        return self.w[k]

    cdef inline void update_active(self, int k, double g):
        cdef double d, z, sq
        self.q[k] += g**2
        sq = sqrt(self.q[k])
        if self.L == 2:
            self.w[k] = (self.w[k]*sq - self.eta*g)/(self.eta*self.C + sq)
        elif self.L == 1:
            z = self.w[k] - self.eta*g/sq
            d = abs(z) - self.eta*self.C/sq
            self.w[k] = sign(z) * max(0, d)
        else:
            self.w[k] -= self.eta*g/sq
        self.u[k] = self.step+1

