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
from numpy import zeros, empty, exp, log, ones, zeros_like, ones_like
import numpy.random as rand
from arsenal.alphabet import Alphabet

from libc.math cimport log1p, exp, log
from libcpp.pair cimport pair
from libcpp.map cimport map
from libcpp.vector cimport vector

cdef double INF = np.inf
cdef double NINF = -np.inf

cdef inline double logaddexp(double x, double y) nogil:
    """
    Needs to be rewritten
    """
    cdef double tmp = x - y
    if tmp > 0:
        return x + log1p(exp(-tmp))
    elif tmp <= 0:
        return y + log1p(exp(tmp))
    else:
        return x + y

cdef class ChunkSegmenter(object):

    cdef int K
    cdef words, segments
    
    def __init__(self, K=3):
        # number of states
        # Note: this is hard-coded to three: prefix, stem, suffix,
        # but could be extended to a larger set as in Cotterell et al. (2015).
        # See http://www.aclweb.org/anthology/K/K15/K15-1017.pdf for more details.
        # 0 = prefix, 1 = stem, 2 = suffix
        self.K = K

        # word dictionary
        self.words = Alphabet()
        # segment dictionary
        self.segments = Alphabet()

    cpdef double logZ(self, double[:, :, :] ge, double[:] gt0, double[:, :] gt, int M):
        """ Computes the log partition function """
        cdef double[:, :] b = empty((M+1, self.K))
        self.backward(b, ge, gt0, gt, M)
        return b[0, 0]

    cdef void forward(self, double[:, :] a, double[:, :, :] ge, double[:] gt0, double[:, :] gt, int M):
        """
        Forward algorithm to compute the alpha values and the risk
        under the correct setting of the latent variables and the parameters.
        """
        cdef int K = self.K
        cdef int tp, t, y

        for y in xrange(K):
            for t in xrange(M+1):
                a[t, y] = NINF
        a[0, 0] = 0.0

        for tp in xrange(M+1):
            for t in xrange(tp+1, M+1):
                for y in xrange(K):
                    if tp == 0:
                        a[t, y] = logaddexp(a[t, y], ge[y, tp, t] + gt0[y])
                    for yp in xrange(K):
                        if tp > 0:
                            a[t, y] = logaddexp(a[t, y], a[tp, yp] + ge[y, tp, t] + gt[yp, y])                            

    cdef void backward(self, double[:, :] b, double[:, :, :] ge, double[:] gt0, double[:, :] gt, int M):
        """ Backward algorithm for a semi-Markov model. """

        cdef int K = self.K
        cdef int t, tp, y, segid
        
        for y in xrange(K):
            for t in xrange(M+1):
                b[t, y] = NINF
            b[M, y] = 0.0
            
        for t in xrange(M, 0, -1):
            for tp in xrange(t-1, -1, -1):
                for y in xrange(K):
                    for yp in xrange(K):
                        if tp == 0:
                            b[tp, yp] = logaddexp(b[tp, yp], b[t, y] + ge[y, tp, t] + gt0[y])
                        else:
                            b[tp, yp] = logaddexp(b[tp, yp], b[t, y] + ge[y, tp, t] + gt[yp, y])

    cpdef object decode(self, object x, double[:, :, :] ge, double[:] gt0, double[:, :] gt):
        """ Decodes the model using the Viterbi algorithm """
        cdef int M = x.M
        cdef int K = self.K
        cdef double[:, :] g = empty((M+1, K))
        cdef int[:, :] bp_t = empty((M+1, K), dtype=np.int32)
        cdef int[:, :] bp_y = empty((M+1, K), dtype=np.int32)
        cdef double best = 0.0
        
        best, s_list, labels = self.viterbi(g, bp_t, bp_y, ge, gt0, gt, M)
        labels.reverse()
        segments = []
        tp = 0
        for t in reversed(s_list):
            segments.append((tp, t))
            tp = t
        return best, segments, labels
    
    cdef object viterbi(self, double[:, :] g, int[:, :] bp_t, int[:, :] bp_y, double[:, :, :] ge, double[:] gt0, double[:, :] gt, int M):
        """
        Forward algorithm in the tropical semiring compute the gamma values and 
        then get the proper back pointers..
        """
        cdef int K = self.K
        cdef int tp, t, y, y_best, y_cur, t_cur, tmp
        cdef double score, best
        
        for y in xrange(K):
            for t in xrange(M+1):
                g[t, y] = NINF
                bp_t[t, y] = 0
                bp_y[t, y] = 0
        g[0, 0] = 0.0
        best = NINF

        # viterbi algorithm
        for tp in xrange(M+1):
            for t in xrange(tp+1, M+1):
                for y in xrange(K):
                    if tp == 0:
                        score = g[tp, 0] + ge[y, tp, t] + gt0[y]
                        if score >= g[t, y]:
                            g[t, y] = score
                            bp_t[t, y] = 0
                            bp_y[t, y] = 0
                    for yp in xrange(K):
                        if tp > 0:
                            score = g[tp, yp] + ge[y, tp, t] + gt[yp, y]
                            if score >= g[t, y]:
                                g[t, y] = score
                                bp_t[t, y] = tp
                                bp_y[t, y] = yp

        # back pointers
        cdef vector[int] segments = vector[int]()
        cdef vector[int] labels = vector[int]()

        for y in xrange(K):
            if g[M, y] >= best:
                best = g[M, y]
                y_best = y

        y_cur, t_cur = y_best, M
        while t_cur > 0:
            segments.push_back(t_cur)
            labels.push_back(y_cur)
            tmp = bp_y[t_cur, y_cur]
            t_cur = bp_t[t_cur, y_cur]
            y_cur = tmp
            
        return best, segments, labels
                            
    cdef void dlogZ(self, double[:, :, :] ge, double[:] gt0, double[:, :] gt, double[:, :, :] gge, double[:] ggt0, double[:, :] ggt, int M):
        """ gradient of the parameters """

        cdef int y, yp, tp, t, segid
        cdef int K = self.K
        cdef double score
        cdef double[:, :] a = empty((M+1, self.K))
        cdef double[:, :] b = empty((M+1, self.K))
        self.forward(a, ge, gt0, gt, M)
        self.backward(b, ge, gt0, gt, M)
        cdef double logZ = b[0, 0]
        for tp in xrange(M+1):
            for t in xrange(tp+1, M+1):
                for y in xrange(K):
                    if tp == 0:
                        score = exp(a[tp, 0] + b[t, y] + ge[y, tp, t] + gt0[y] - logZ)
                        gge[y, tp, t] += score
                        ggt0[y] += score

                    for yp in xrange(K):
                        if tp > 0:
                            score = exp(a[tp, yp] + b[t, y] + ge[y, tp, t] + gt[yp, y] - logZ)
                            gge[y, tp, t] += score
                            ggt[yp, y] += score
                            
    cpdef double ll(self, object x, double[:, :, :] ge, double[:] gt0, double[:, :] gt):
        """ Log-likelihood """
        cdef int M = x.M
        
        # clamped inference
        cdef int y, yp, t, tp
        cdef double clamped = 0.0
        yp = -1
        for y, (tp, t) in zip(x.lab, x.indices):
            if tp == 0:
                clamped += gt0[y]
                clamped += ge[y, tp, t]
            else:
                clamped += gt[yp, y]
                clamped += ge[y, tp, t]
            yp = y
        return clamped - self.logZ(ge, gt0, gt, M)

    cpdef void dll(self, object x, double[:, :, :] ge, double[:] gt0, double[:, :] gt, double[:, :, :] gge, double[:] ggt0, double[:, :] ggt):
        """ Computes the gradient of the log-likelihood """
        cdef int M = x.M

        # clamped inference
        cdef int y, yp, t, tp
        yp = -1
        for y, (tp, t) in zip(x.index_labels, x.indices):
            if tp == 0:
                ggt0[y] -= 1.0
                gge[y, tp, t] -= 1.0
            else:
                ggt[yp, y] -= 1.0
                gge[y, tp, t] -= 1.0
            yp = y
        # unclamed inference
        self.dlogZ(ge, gt0, gt, gge, ggt0, ggt, M)

    def enumerate(self, M):
        """ 
        Enumerate all paths through the semi-CRF. 
        The implementation here runs the semi-Markov
        forward algorithm over the derivation semiring (Goodman 1999).
        See http://www.aclweb.org/anthology/J99-4004 for the concept
        applied to CKY parsing.
        """

        K = self.K
        # set up the table
        paths = empty((M+1, K), dtype=object)
        for t in xrange(M+1):
            for y in xrange(K):
                paths[t, y] = []

        # initial paths are empty
        for y in xrange(K):
            paths[0, y] = [[]]

        # run the dynamic program
        for tp in xrange(M+1):
            for t in xrange(tp+1, M+1):
                for yp in xrange(K):
                    for y in xrange(K):
                        if tp > 0 or yp == 0:
                            paths[t, y] += [path + [(tp, t, y)] for path in paths[tp, yp]]

        return paths[M].sum()
    
    def enumerate_logZ(self, ge, gt0, gt, M):
        " Brute Force the partition function Z"

        K = self.K
        # gt0 = zeros((K))
        # gt = zeros((K, K))
        # ge = zeros((K, N, N+1))
        # self.log_potentials(word, theta, gt0, gt, ge, True)

        Z = 0.0
        for path in self.enumerate(M):
            score = 1.0
            # initialization
            arc = path[0]
            tp, t, y = arc[0], arc[1], arc[2]
            score *= exp(ge[y, tp, t] + gt0[y])
            for i, arc in enumerate(path[1:]):
                yp = path[i][2]
                tp, t = arc[0], arc[1]
                y = arc[2]
                score *= exp(ge[y, tp, t] + gt[yp, y])
            Z += score

        return log(Z)

    def enumerate_max(self, ge, gt0, gt, M):
        " Brute Force the Viterbi segmentation"

        K = self.K
        best = NINF
        best_segments, best_labels = [], []
        for path in self.enumerate(M):
            segments, labels = [], []
            score = 1.0
            # initialization
            arc = path[0]
            tp, t, y = arc[0], arc[1], arc[2]
            segments.append((tp, t))
            labels.append(y)
            score *= exp(ge[y, tp, t] + gt0[y])
            for i, arc in enumerate(path[1:]):
                yp = path[i][2]
                tp, t = arc[0], arc[1]
                segments.append((tp, t))
                y = arc[2]
                labels.append(y)
                score *= exp(ge[y, tp, t] + gt[yp, y])

            if score >= best:
                best = score
                best_labels = labels
                best_segments = segments
        return log(best), best_segments, best_labels

