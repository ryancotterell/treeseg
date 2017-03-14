#!/usr/bin/env python
# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: profile=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = ["-std=c++11"]

import numpy as np
from numpy import zeros, ones, zeros_like, ones_like

from libc.math cimport log1p, exp, log
from libcpp.pair cimport pair
from libcpp.vector cimport vector

from segmenter.data.datastructures cimport Tree

cdef double INF = np.inf
cdef double NINF = -np.inf

cdef inline double logaddexp(double x, double y) nogil:
    """ TODO: switch to log table """
    cdef double tmp = x - y
    if tmp > 0:
        return x + log1p(exp(-tmp))
    elif tmp <= 0:
        return y + log1p(exp(tmp))
    else:
        return x + y

cdef class TreeSegmenter(object):
    """ Tree segmenter with inference based on CKY """

    cdef readonly int G
    
    def __init__(self, G=3):
        self.G = G

    cpdef double ll(self, Tree tree, double[:, :, :] psi):
        """ log-likelihood """
        return -self.score(tree, psi) + self.logZ(tree.M, self.G, psi)

    cpdef double[:, :, :] dll(self, Tree tree, double[:, :, :] psi):
        """ gradient of the log-liklihood """
        cdef double[:, :, :] dpsi = zeros_like(psi)
        self._dscore(tree, psi, dpsi)
        self._dlogZ(tree.M, self.G, psi, dpsi)
        return dpsi
        
    cpdef double score(self, Tree tree, double[:, :, :] psi):
        """ scores a tree w.r.t. the current parameters """
        cdef int index, i, j, l
        cdef pair[int, int] ij
        cdef double score = 0.0
        for index in xrange(tree.size):
            ij = tree.spans[index]
            i, j = ij.first, ij.second
            l = tree.labels[index]
            score += psi[i, j, l]
        return score

    cpdef double[:, :, :] dscore(self, Tree tree, double[:, :, :] psi):
        """ dscore from Python """
        cdef double[:, :, :] dpsi = zeros_like(psi)
        self._dscore(tree, psi, dpsi)
        return dpsi
    
    cdef void _dscore(self, Tree tree, double[:, :, :] psi, double[:, :, :] dpsi):
        """ computes the gradient of a tree (linear function) w.r.t. the current parameters """
        cdef int index, i, j, l
        cdef pair[int, int] ij
        for index in xrange(tree.size):
            ij = tree.spans[index]
            i, j = ij.first, ij.second
            l = tree.labels[index]
            dpsi[i, j, l] -= 1.0

    cpdef double logZ(self, int M, int G, double[:, :, :] psi):
        """ log partition function """
        cdef double[:, :, :] a = ones_like(psi)*NINF
        self.inside(M, G, psi, a)
        return a[0, M, 0]
        
    cdef void inside(self, int N, int G, double[:, :, :] psi, double[:, :, :] a):
        """ inside algorithm """
        cdef int i, j, k, l, g
        # precompute the segmentation scores
        for i in xrange(N+1):
            for j in xrange(i+1, N+1):
                for g in xrange(G):
                    a[i, j, g] = psi[i, j, g]
        # CKY (inside algorithm) of this specific grammar
        for l in xrange(2, N+1): # length of span
            for i in xrange(N-l+1):
                for j in xrange(i+1, i+l):
                    k = i + l
                    for g1 in xrange(1, G):
                        for g2 in xrange(1, G):
                            a[i, k, 0] = logaddexp(a[i, k, 0], a[i, j, g1] + a[j, k, g2] + psi[i, k, 0])
                    for g in xrange(1, G):
                        a[i, k, 0] = logaddexp(a[i, k, 0], a[i, j, 0] + a[j, k, g] + psi[i, k, 0])
                        a[i, k, 0] = logaddexp(a[i, k, 0], a[i, j, g] + a[j, k, 0] + psi[i, k, 0])
                    a[i, k, 0] = logaddexp(a[i, k, 0], a[i, j, 0] + a[j, k, 0] + psi[i, k, 0])

    cpdef double[:, :, :] dlogZ(self, int M, int G, double[:, :, :] psi):
        """ dlogZ visible from Python """
        cdef double[:, :, :] a = ones_like(psi)*NINF
        self.inside(M, G, psi, a)
        cdef double[:, :, :] da = ones_like(psi)*NINF
        da[0, M, 0] = 0.0
        cdef double[:, :, :] dpsi = zeros_like(psi)
        self.outside(M, G, psi, dpsi, a, da)
        return dpsi

    cdef void _dlogZ(self, int M, int G, double[:, :, :] psi, double[:, :, :] dpsi):
        """ dlogZ """
        cdef double[:, :, :] a = ones_like(psi)*NINF
        self.inside(M, G, psi, a)
        cdef double[:, :, :] da = ones_like(psi)*NINF
        da[0, M, 0] = 0.0
        self.outside(M, G, psi, dpsi, a, da)

    cdef void outside(self, int M, int G, double[:, :, :] psi, double[:, :, :] dpsi, double[:, :, :] a, double[:, :, :] da):
        """ outside algorithm """    
        cdef double logZ = a[0, M, 0]
        cdef int i, j, k, l
        # CKY (outside algorithm) of this specific grammar
        for l in reversed(xrange(2, M+1)): # length of span
            for i in reversed(xrange(M-l+1)):
                for j in reversed(xrange(i+1, i+l)):
                    k = i + l
                    da[i, j, 0] = logaddexp(da[i, j, 0], da[i, k, 0] + a[j, k, 0] + psi[i, k, 0])
                    da[j, k, 0] = logaddexp(da[j, k, 0], da[i, k, 0] + a[i, j, 0] + psi[i, k, 0])
                    dpsi[i, k, 0] += exp(da[i, k, 0] + a[i, j, 0] + a[j, k, 0] + psi[i, k, 0] - logZ)
                    for g in xrange(1, G):
                        da[i, j, g] = logaddexp(da[i, j, g], a[j, k, 0] + da[i, k, 0] + psi[i, k, 0])
                        da[j, k, 0] = logaddexp(da[j, k, 0], a[i, j, g] + da[i, k, 0] + psi[i, k, 0])
                        dpsi[i, k, 0] += exp(da[i, k, 0] + a[i, j, g] + a[j, k, 0] + psi[i, k, 0] - logZ)

                        da[i, j, 0] = logaddexp(da[i, j, 0], da[i, k, 0] + a[j, k, g] + psi[i, k, 0])
                        da[j, k, g] = logaddexp(da[j, k, g], da[i, k, 0] + a[i, j, 0] + psi[i, k, 0])
                        dpsi[i, k, 0] += exp(da[i, k, 0] + a[i, j, 0] + a[j, k, g] + psi[i, k, 0] - logZ)
                    for g1 in xrange(1, G):
                        for g2 in xrange(1, G):
                            da[i, j, g1] = logaddexp(da[i, j, g1], da[i, k, 0] + a[j, k, g2] + psi[i, k, 0])
                            da[j, k, g2] = logaddexp(da[j, k, g2], da[i, k, 0] + a[i, j, g1] + psi[i, k, 0])
                            dpsi[i, k, 0] += exp(da[i, k, 0] + a[i, j, g1] + a[j, k, g2] + psi[i, k, 0] - logZ)
        for i in xrange(M+1):
             for j in xrange(i+1, M+1):
                 for g in xrange(G):
                     dpsi[i, j, g] += exp(psi[i, j, g] + da[i, j, g] - logZ)

    cdef void viterbi(self, int M, int G, double[:, :, :] psi, double[:, :, :] a, int[:, :, :, :] bp):
        """ inside algorithm """

        # precompute the segmentation scores
        cdef int i, j, k, l, g
        cdef double score 
        for i in xrange(M+1):
            for j in xrange(i+1, M+1):
                for g in xrange(G):
                    a[i, j, g] = psi[i, j, g]

        # CKY (inside algorithm) of this specific grammar
        for l in xrange(2, M+1): # length of span
            for i in xrange(M-l+1):
                for j in xrange(i+1, i+l):
                    k = i + l
                    for g1 in xrange(1, G):
                        for g2 in xrange(1, G):
                            score = a[i, j, g1] + a[j, k, g2] + psi[i, k, 0]
                            if score >= a[i, k, 0]:
                                a[i, k, 0] = score
                                bp[i, k, 0, 0] = j
                                bp[i, k, 0, 1] = g1
                                bp[i, k, 0, 2] = g2
                                
                    for g in xrange(1, G):
                        score = a[i, j, 0] + a[j, k, g] + psi[i, k, 0]
                        if score >= a[i, k, 0]:
                            a[i, k, 0] = score
                            bp[i, k, 0, 0] = j
                            bp[i, k, 0, 1] = 0
                            bp[i, k, 0, 2] = g

                        score = a[i, j, g] + a[j, k, 0] + psi[i, k, 0]
                        if score >= a[i, k, 0]:
                            a[i, k, 0] = score
                            bp[i, k, 0, 0] = j
                            bp[i, k, 0, 1] = g
                            bp[i, k, 0, 2] = 0

                    score = a[i, j, 0] + a[j, k, 0] + psi[i, k, 0]
                    if score >= a[i, k, 0]:
                        a[i, k, 0] = score
                        bp[i, k, 0, 0] = j
                        bp[i, k, 0, 1] = 0
                        bp[i, k, 0, 2] = 0
    
    def argmax(self, M_, G_, psi_, string):
        """ decode """

        cdef int i, j, k, g, g1, g2
        cdef int M = M_
        cdef int G = G_
        cdef double[:, :, :] psi = psi_
        cdef double[:, :, :] a = zeros((M+1, M+1, G))
        cdef int[:, :, :, :] bp = ones((M+1, M+1, G, 3), dtype=np.int32)*-1
        self.viterbi(M, G, psi, a, bp)
        spans = []
        
        def extract(i, k, g, string):
            spans.append((i, k))

            j = bp[i, k, g, 0]
            g1 = bp[i, k, g, 1]
            g2 = bp[i, k, g, 2]
            if j == -1:
                return string[i:k] + ":" + str(g)

            result = "( "
            result += extract(i, j, g1, string)
            result += " "
            result += extract(j, k, g2, string)
            result += " )"
            return result

        return a[0, M, 0], extract(0, M, 0, string), spans

