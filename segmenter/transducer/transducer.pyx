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
from numpy import zeros, ones, empty, ones, zeros_like, ones_like, asarray, int32
from numpy import exp as npexp
from numpy import exp as nplog

import numpy.random as rand
import itertools as it
from arsenal.alphabet import Alphabet

from libc.math cimport log1p, exp, log
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libc.stdlib cimport rand, RAND_MAX

cdef double INF = np.inf
cdef double NINF = -np.inf
cdef int INT_INF = 1000000

cdef inline double logaddexp(double x, double y) nogil:
    """ log(exp(x) + exp(y)), but numerically stable """
    # TODO: Needs to be rewritten
    cdef double tmp = x - y
    if tmp > 0:
        return x + log1p(exp(-tmp))
    elif tmp <= 0:
        return y + log1p(exp(tmp))
    else:
        return x + y

cdef class Transducer(object):
    """ 
    CRF Transducer that allows n-to-m alignments and 
    scores variable-order lengths of output context.

    TODO: implement failure arcs
    """

    cdef readonly int[:, :] A, CONCAT
    cdef readonly int[:] NEXT, LENX, LENY
    cdef readonly int IL, lenP, lenX, lenY
    cdef readonly object Sigma, X, Y, P, Z
    
    def __init__(self, Sigma, X, Y, P, Z, IL=2):
        self.IL = IL
        self.Sigma = Sigma
        self.X = X
        self.Y = Y
        self.P = P
        self.Z = Z

        self.A = ones((len(X), len(Y)), dtype=int32)
        self.A[0, 0] = -1
        self.lenP = len(self.P)
        self.lenX = len(self.X)
        self.lenY = len(self.Y)
        
        # zero must be mapped to the empty string
        assert self.P[""] == 0
        assert self.X[""] == 0
        assert self.Y[""] == 0
        assert self.Sigma[""] == 0

        self.LENX = zeros((len(self.X)), dtype=int32)
        self.LENY = zeros((len(self.Y)), dtype=int32)
        self.NEXT = zeros((len(self.Z)), dtype=int32)
        self.CONCAT = zeros((len(self.P), len(self.Y)), dtype=int32)
        
        # get the length
        cdef int xi
        for x, xi in self.X.items():
            self.LENX[xi] = len(x)
        cdef int yi
        for y, yi in self.Y.items():
            self.LENY[yi] = len(y)

        # make sure every element of Sigma can be inserted
        for s in self.Sigma:
            assert s in self.Y 

        # get next transduction
        self.next()
        
    def next(self):
        """ create the next mapping """
        cdef int pi, yi, z
        for p, pi in self.P.items():
            for y, yi in self.Y.items():
                py = p+y
                z = self.Z[py]
                self.CONCAT[pi, yi] = z
                # suffixes
                for i in xrange(len(py)+1):
                    pp = py[i:]
                    if pp in self.P:
                        self.NEXT[z] = self.P[pp]
                        break

    def score(self, string1, string2, a, psi):
        """ scores an alignment """
        i, k = 0, 0
        score = 1.0
        for (xx, yy) in a:
            x, y = self.X[xx], self.Y[yy]
            p, pp, kp = 0, "", 0
            for kp in xrange(k):
                pp = string2[kp:k]
                if pp in self.P:
                    p = self.P[pp]
                    break
            score *= exp(psi[i, x, y, p])
            i += len(xx)
            k += len(yy)
        return score
    
    def enum_clamped(self, string1, string2, Sigma, IL, X, Y, psi):
        """ clamped by enumeration """
        clamped = 0.0
        for a in self.enumerate_alignments(string1, string2, X, Y):
            a.reverse()
            s = self.score(string1, string2, a, psi)
            clamped += s
        return log(clamped)

    def enum_logZ(self, string1, Sigma, IL, X, Y, psi):
        """ logZ by enumeration """
        Z = 0.0
        for string2 in self.enumerate_strings(string1, Sigma, IL):
            for a in self.enumerate_alignments(string1, string2, X, Y):
                a.reverse()
                s = self.score(string1, string2, a, psi)
                Z += s
        return log(Z)
    
    def enum_max(self, string1, Sigma, IL, X, Y, psi):
        """ max by enumeration """
        max_score, max_a = NINF, None
        for string2 in self.enumerate_strings(string1, Sigma, IL):
            for a in self.enumerate_alignments(string1, string2, X, Y):
                a.reverse()
                score = self.score(string1, string2, a, psi)
                if score >= max_score:
                    max_score = score
                    max_a = a
        return log(max_score), max_a

    def enumerate(self, string1, Sigma, IL, X, Y):
        """ enumerates over the support """
        A = []
        for string2 in self.enumerate_strings(string1, Sigma, IL):
            for a in self.enumerate_alignments(string1, string2, X, Y):
                a.reverse()
                A.append(a)
        return A
    
    def enumerate_strings(self, string1, Sigma, IL):
        """ Enumerate all output strings. """
        strings = set([])
        N1 = len(string1)
        lst = list(Sigma)
        lst.remove("")
        for n in xrange(N1+IL+1):
            for tup in it.product(lst, repeat=n):
                string2 = "".join(tup)
                strings.add(string2)
        return strings
                
    def enumerate_alignments(self, string1, string2, X, Y):
        """ #numerate all alignments between string1 and string2. """
        N1, N2 = len(string1), len(string2)
        d = empty((N1+1, N2+1), dtype='object')
        for i in xrange(N1+1):
            for j in xrange(N2+1):
                d[i, j] = []
        d[0, 0] = [[]]
        for i in xrange(N1+1):
            for j in xrange(i, N1+1):
                seg1 = string1[i:j]
                if seg1 not in X:
                    continue
                for k in xrange(N2+1):
                    for l in xrange(k, N2+1):
                        seg2 = string2[k:l]
                        if seg1 == "" and seg2 == "": # no eps-eps
                            continue
                        if seg2 not in Y:
                            continue
                        a = (seg1, seg2) # alignment
                        for plst in d[i, k]:
                            lst = [a] + plst
                            d[j, l].append(lst)
        return d[-1, -1]

    cdef void forward_clamped(self, int N, int [:, :] X, int M, int[:, :] Y, double[:, :, :] a, \
                              double[:, :, :, :] psi, int minx=INT_INF, int miny=INT_INF, int maxoffset=INT_INF):
        """ forward algorithm for clamped inference"""
        # assign variables
        cdef int lenP = self.lenP
        cdef int lenY = self.lenY
        cdef int[:, :] CONCAT = self.CONCAT
        cdef int[:, :] A = self.A
        cdef int[:] NEXT = self.NEXT
        cdef int[:] LENY = self.LENY
        cdef int i, j, k, l, x, y
        cdef int p, n, z

        # core algorithm
        for i in xrange(N + 1):
            for j in xrange(i, min(i + minx, N + 1)):
                x = X[i, j]
                if x == -1:                                                       # skip illicit input segments
                    continue
                for k in xrange(M + 1):
                    if abs(i - k) >= maxoffset:                                   # enforce constrained monotonicity
                        continue
                    for l in xrange(k, min(k + miny, M + 1)):
                        y = Y[k, l]
                        if y == -1:
                            continue
                        if A[x, y] == -1:                                         # enforce alignment dictionary
                            continue
                        if LENY[y] > miny:
                            continue
                        if abs(j - l) >= maxoffset:                               # enforce constrained monotonicity
                            continue
                        for p in xrange(lenP):                                    # previous output Markov order
                            z = CONCAT[p, y]
                            n = NEXT[z]
                            a[j, l, n] = logaddexp(a[j, l, n], \
                                                   psi[i, x, y, p] + a[i, k, p])  # dynamic program
                            
    cdef void backward_clamped(self, int N, int [:, :] X, int M, int[:, :] Y, double[:, :, :] b, \
                               double[:, :, :, :] psi, int minx=INT_INF, int miny=INT_INF, int maxoffset=INT_INF):
        """ backward algorithm """
        # assign variables
        cdef int lenP = self.lenP
        cdef int lenY = self.lenY
        cdef int[:, :] CONCAT = self.CONCAT
        cdef int[:, :] A = self.A
        cdef int[:] NEXT = self.NEXT
        cdef int[:] LENY = self.LENY
        cdef int i, j, k, l, x, y
        cdef int p, n, z
        
        # core algorithm
        for i in reversed(xrange(N + 1)):
            for j in reversed(xrange(i, min(i + minx, N + 1))):
                x = X[i, j]
                if x == -1:                                                       # skip illicit input segments
                    continue
                for k in reversed(xrange(M + 1)):
                    if abs(i - k) >= maxoffset:                                   # enforce constrained monotonicity
                        continue
                    for l in reversed(xrange(k, min(k + miny, M + 1))):
                        y = Y[k, l]
                        if y == -1:
                            continue
                        if A[x, y] == -1:                                         # enforce alignment dictionary
                            continue
                        if LENY[y] > miny:
                            continue
                        if abs(j - l) >= maxoffset:                               # enforce constrained monotonicity
                            continue
                        for p in xrange(lenP):                                    # previous output Markov order
                            z = CONCAT[p, y]
                            n = NEXT[z]
                            b[i, k, p] = logaddexp(b[i, k, p], \
                                                   psi[i, x, y, p] + b[j, l, n])  # dynamic program

    cdef void _dclamped(self, int N, int [:, :] X, int M, int[:, :] Y, double[:, :, :] a, double[:, :, :] b, \
                        double[:, :, :, :] psi, double[:, :, :, :] dpsi, int minx=INT_INF, int miny=INT_INF, int maxoffset=INT_INF):
        """ forward algorithm for clamped inference"""
        # assign variables
        cdef int lenP = self.lenP
        cdef int lenY = self.lenY
        cdef int[:, :] CONCAT = self.CONCAT
        cdef int[:, :] A = self.A
        cdef int[:] NEXT = self.NEXT
        cdef int[:] LENY = self.LENY
        cdef int i, j, k, l, x, y
        cdef int p, n, z
        cdef double logZ = b[0, 0, 0]
        
        # core algorithm
        for i in reversed(xrange(N + 1)):
            for j in reversed(xrange(i, min(i + minx, N + 1))):
                x = X[i, j]
                if x == -1:                                                       # skip illicit input segments
                     continue
                for k in reversed(xrange(M + 1)):
                     if abs(i - k) >= maxoffset:                                   # enforce constrained monotonicity
                         continue
                     for l in reversed(xrange(k, min(k + miny, M + 1))):
                         y = Y[k, l]
                         if y == -1:
                             continue
                         if A[x, y] == -1:                                         # enforce alignment dictionary
                             continue
                         if LENY[y] > miny:
                             continue
                         if abs(j - l) >= maxoffset:                               # enforce constrained monotonicity
                             continue
                         for p in xrange(lenP):                                    # previous output Markov order
                             z = CONCAT[p, y]
                             n = NEXT[z]
                             dpsi[i, x, y, p] += exp(a[i, k, p] + b[j, l, n] + psi[i, x, y, p] - logZ)

    def clamped(self, string1, string2, psi, minx=INT_INF, miny=INT_INF, maxoffset=INT_INF):
        """ clamped function """
        N, M, IL = len(string1), len(string2), self.IL
        X = self.extractX(string1, minx, miny, maxoffset)
        Y = self.extractY(string2, minx, miny, maxoffset)

        b = ones((N+1, M+1, self.lenP)) * NINF
        b[-1, -1, :] = 0.0
        self.backward_clamped(N, X, M, Y, b, psi, minx, miny, maxoffset)
        return b[0, 0, 0]

    def dclamped(self, string1, string2, psi, minx=INT_INF, miny=INT_INF, maxoffset=INT_INF):
        """ gradient of the clamp function """
        N, M, IL = len(string1), len(string2), self.IL
        X = self.extractX(string1, minx, miny, maxoffset)
        Y = self.extractY(string2, minx, miny, maxoffset)

        a = ones((N+1, M+1, self.lenP)) * NINF
        a[0, 0, 0] = 0.0
        self.forward_clamped(N, X, M, Y, a, psi, minx, miny, maxoffset)
        b = ones((N+1, M+1, self.lenP)) * NINF
        b[-1, -1, :] = 0.0
        self.backward_clamped(N, X, M, Y, b, psi, minx, miny, maxoffset)
        
        dpsi = zeros_like(psi)
        self._dclamped(N, X, M, Y, a, b, psi, dpsi, minx, miny, maxoffset)
        return dpsi
                            
    cdef void forward(self, int N, int [:, :] X, double[:, :, :] a, \
                      double[:, :, :, :] psi, int minx=INT_INF, int miny=INT_INF, int maxoffset=INT_INF):
        """ forward algorithm """
        # assign variables
        cdef int lenP = self.lenP
        cdef int lenY = self.lenY
        cdef int[:, :] CONCAT = self.CONCAT
        cdef int[:, :] A = self.A
        cdef int[:] NEXT = self.NEXT
        cdef int[:] LENY = self.LENY
        cdef int IL = self.IL
        cdef int i, j, k, l, x, y
        cdef int p, n, z
        
        # core algorithm
        for i in xrange(N + 1):
            for j in xrange(i, min(i + minx, N + 1)):
                x = X[i, j]
                if x == -1:                                                       # skip illicit input segments
                    continue
                for k in xrange(N + 1 + IL):
                    if abs(i - k) >= maxoffset:                                   # enforce constrained monotonicity
                        continue
                    for y in xrange(lenY):
                        if A[x, y] == -1:                                         # enforce alignment dictionary
                            continue
                        if LENY[y] > miny:
                            continue
                        l = k + LENY[y]
                        if l > N + IL:                                            # cannot go beyond the insertion limit
                            continue
                        if abs(j - l) >= maxoffset:                               # enforce constrained monotonicity
                            continue
                        for p in xrange(lenP):                                    # previous output Markov order
                            z = CONCAT[p, y]
                            n = NEXT[z]
                            a[j, l, n] = logaddexp(a[j, l, n], \
                                                   psi[i, x, y, p] + a[i, k, p])  # dynamic program

    cdef void backward(self, int N, int [:, :] X, double[:, :, :] b, \
                       double[:, :, :, :] psi, int minx=INT_INF, int miny=INT_INF, int maxoffset=INT_INF):
        """ backward algorithm """
        # assign variables
        cdef int lenP = self.lenP
        cdef int lenY = self.lenY
        cdef int[:, :] CONCAT = self.CONCAT
        cdef int[:, :] A = self.A
        cdef int[:] NEXT = self.NEXT
        cdef int[:] LENY = self.LENY
        cdef int IL = self.IL
        cdef int i, j, k, l, x, y
        cdef int p, n, z
        
        # core algorithm
        for i in reversed(xrange(N + 1)):
            for j in reversed(xrange(i, min(i + minx, N + 1))):
                x = X[i, j]
                if x == -1:                                                       # skip illicit input segments
                    continue
                for k in reversed(xrange(N + 1 + IL)):
                    if abs(i - k) >= maxoffset:                                   # enforce constrained monotonicity
                        continue
                    for y in xrange(lenY):
                        if A[x, y] == -1:                                         # enforce alignment dictionary
                            continue
                        if LENY[y] > miny:
                            continue
                        l = k + LENY[y]
                        if l > N + IL:                                            # cannot go beyond the insertion limit
                            continue
                        if abs(j - l) >= maxoffset:                               # enforce constrained monotonicity
                            continue
                        for p in xrange(lenP):                                    # previous output Markov order
                            z = CONCAT[p, y]
                            n = NEXT[z]
                            b[i, k, p] = logaddexp(b[i, k, p], \
                                                   psi[i, x, y, p] + b[j, l, n])  # dynamic program

    cdef void local(self, int N, int [:, :] X, double[:, :, :] b, double[:, :, :, :] psi, \
                    double[:, :, :, :, :] psi_local, int minx=INT_INF, int miny=INT_INF, int maxoffset=INT_INF):
        """ locally renormalize  """
        # assign variables
        cdef int lenP = self.lenP
        cdef int lenY = self.lenY
        cdef int[:, :] CONCAT = self.CONCAT
        cdef int[:, :] A = self.A
        cdef int[:] NEXT = self.NEXT
        cdef int[:] LENY = self.LENY
        cdef int IL = self.IL
        cdef int i, j, k, l, x, y
        cdef int p, n, z
        cdef double loglocalZ
        
        # core algorithm
        for i in xrange(N+1):
            for k in xrange(N+IL+1):
                for p in xrange(lenP):
                    loglocalZ = NINF
                    for j in xrange(i, min(i + minx, N + 1)):
                        x = X[i, j]
                        if x == -1:                                              
                            continue
                        for y in xrange(lenY):
                            if A[x, y] == -1:                                         
                                continue
                            if LENY[y] > miny:
                                continue
                            l = k + self.LENY[y]
                            if l > N + IL: # cannot go beyond the insertion limit
                                continue
                            z = CONCAT[p, y]
                            n = NEXT[z]
                            loglocalZ = logaddexp(loglocalZ, psi[i, x, y, p] + b[j, l, n])
                            
                    if i == N: # unweighted final state
                        loglocalZ = logaddexp(loglocalZ, 0.0)
                    for j in xrange(i, min(i + minx, N + 1)):
                        x = X[i, j]
                        if x == -1:                                              
                            continue
                        for y in xrange(lenY):
                            if A[x, y] == -1:                                         
                                continue
                            if LENY[y] > miny:
                                continue
                            l = k + self.LENY[y]
                            if l > N + IL: # cannot go beyond the insertion limit
                                continue
                            z = CONCAT[p, y]
                            n = NEXT[z]
                            psi_local[i, k, x, y, p] = exp(psi[i, x, y, p] + b[j, l, n] - loglocalZ)
                            
    def local_renormalize(self, string1, psi, minx=INT_INF, miny=INT_INF, maxoffset=INT_INF):
        """ locally renormalize the model """
        N, IL = len(string1), self.IL
        X = self.extractX(string1, minx, miny, maxoffset)
        b = ones((N+1, N+IL+1, self.lenP)) * NINF
        b[-1, :, :] = 0.0
        self.backward(N, X, b, psi, minx, miny, maxoffset)


        cdef double[:, :, :, :, :] psi_local = zeros((N+1, N+IL+1, len(self.X), len(self.Y), len(self.P)))
        self.local(N, X, b, psi, psi_local, minx, miny, maxoffset)
        return psi_local

    def sample(self, string1, psi, num=1, minx=INT_INF, miny=INT_INF, maxoffset=INT_INF):
        """ sample """

        cdef int n, p
        cdef int lenP = self.lenP
        cdef int lenX = self.lenX
        cdef int lenY = self.lenY
        cdef int N = len(string1)
        cdef int IL = self.IL
        cdef int len_probs = lenX*lenY+1
        cdef double[:] probs = zeros((len_probs))
        
        cdef int[:, :] X = self.extractX(string1, minx, miny, maxoffset)
        cdef double[:, :, :] b = ones((N+1, N+IL+1, self.lenP)) * NINF
        for p in xrange(self.lenP):
            for n in xrange(N+IL+1):
                b[N, n, p] = 0.0
        self.backward(N, X, b, psi, minx, miny, maxoffset)
        cdef double[:, :, :, :, :] psi_local = zeros((N+1, N+IL+1, len(self.X), len(self.Y), len(self.P)))
        self.local(N, X, b, psi, psi_local, minx, miny, maxoffset)

        samples = []
        for _ in xrange(num):
            xs, ys = self._sample(N, X, psi_local, probs)
            samples.append([(self.X.lookup(x), self.Y.lookup(y)) for x, y in zip(xs, ys)])
        return samples
   
    cdef pair[vector[int], vector[int]] _sample(self, int N, int[:, :] X, double[:, :, :, :, :] psi_local, double[:] probs, int minx=INT_INF, int miny=INT_INF, int maxoffset=INT_INF):
        """ Take sample from the model using a locally renormalized potential """

        # assign variables
        cdef int lenP = self.lenP
        cdef int lenX = self.lenX
        cdef int lenY = self.lenY
        cdef int len_probs = lenX*lenY+1
        cdef int[:, :] CONCAT = self.CONCAT
        cdef int[:, :] A = self.A
        cdef int[:] NEXT = self.NEXT
        cdef int[:] LENY = self.LENY
        cdef int[:] LENX = self.LENX
        cdef int IL = self.IL
        cdef int i, j, k, x, y, p, z, last, selection
        cdef double r, cum
        cdef double Z = 0.0

        cdef vector[int] xs = vector[int]()
        cdef vector[int] ys = vector[int]()
        
        # core algorithm
        i, j, p = 0, 0, 0
        last = lenX*lenY
        while i < N+1 and j < N+IL+1:
            # constraints
            Z = 0.0
            k = 0
            for x in xrange(lenX):
                for y in xrange(lenY):
                    probs[k] = psi_local[i, j, x, y, p]
                    Z += probs[k]
                    k += 1
            if i == N:
                probs[last] = 1-Z
            # sample next step
            # OLD
            #selection = np.random.choice(last+1, 1, p=probs)[0]
            selection = 0
            cum = 0.0
            r = rand() / float(RAND_MAX)
            for k in xrange(len_probs):
                cum += probs[k]
                if cum >= r:
                    break
                selection += 1
            
            if selection == last:
                break
            else:
                 x = int(selection / lenY)
                 y = selection % lenY
                 xs.push_back(x)
                 ys.push_back(y)
                 i += LENX[x]
                 j += LENY[y]
                 z = CONCAT[p, y]
                 p = NEXT[z]

        return pair[vector[int], vector[int]](xs, ys)
                            
    cdef void _dlogZ(self, int N, int [:, :] X, double[:, :, :] a, double[:, :, :] b, double[:, :, :, :] psi, \
                     double[:, :, :, :] dpsi, int minx=INT_INF, int miny=INT_INF, int maxoffset=INT_INF):
        """ compute dlogZ """
        # assign variables
        cdef int lenP = self.lenP
        cdef int lenY = self.lenY
        cdef int[:, :] CONCAT = self.CONCAT
        cdef int[:, :] A = self.A
        cdef int[:] NEXT = self.NEXT
        cdef int[:] LENY = self.LENY
        cdef int IL = self.IL
        cdef int i, j, k, l, x, y
        cdef int p, n, z
        cdef double logZ = b[0, 0, 0]
        
        # core algorithm
        for i in xrange(N + 1):
            for j in xrange(i, min(i + minx, N + 1)):
                x = X[i, j]
                if x == -1:                                                       # skip illicit input segments
                    continue
                for k in xrange(N + 1 + IL):
                    if abs(i - k) >= maxoffset:                                   # enforce constrained monotonicity
                        continue
                    for y in xrange(lenY):
                        if A[x, y] == -1:                                         # enforce alignment dictionary
                            continue
                        if LENY[y] > miny:
                            continue
                        l = k + LENY[y]
                        if l > N + IL:                                            # cannot go beyond the insertion limit
                            continue
                        if abs(j - l) >= maxoffset:                               # enforce constrained monotonicity
                            continue
                        for p in xrange(lenP):                                    # previous output Markov order
                            z = CONCAT[p, y]
                            n = NEXT[z]
                            dpsi[i, x, y, p] += exp(a[i, k, p] + b[j, l, n] + psi[i, x, y, p] - logZ)
                            
    def logZ(self, string1, psi, minx=INT_INF, miny=INT_INF, maxoffset=INT_INF):
        """ log partition function """
        N, IL = len(string1), self.IL
        X = self.extractX(string1, minx, miny, maxoffset)
        b = ones((N+1, N+IL+1, self.lenP)) * NINF
        b[-1, :, :] = 0.0
        self.backward(N, X, b, psi, minx, miny, maxoffset)
        return b[0, 0, 0]
        
    cpdef double[:, :, :, :] dlogZ(self, object string1, double[:, :, :, :] psi, int minx=INT_INF, int miny=INT_INF, int maxoffset=INT_INF):
        """ compute gradient of the dlogZ """
        cdef int N = len(string1)
        cdef int IL = self.IL
        cdef int[:, :] X = self.extractX(string1, minx, miny, maxoffset)
        
        # run forward and backward
        a = ones((N+1, N+IL+1, self.lenP)) * NINF
        a[0, 0, 0] = 0.0
        b = ones((N+1, N+IL+1, self.lenP)) * NINF
        b[-1, :, :] = 0.0
        self.forward(N, X, a, psi, minx, miny, maxoffset)
        self.backward(N, X, b, psi, minx, miny, maxoffset)

        dpsi = zeros_like(psi)
        self._dlogZ(N, X, a, b, psi, dpsi, minx, miny, maxoffset)
        return dpsi

    def ll(self, string1, string2, psi, minx=INT_INF, miny=INT_INF, maxoffset=INT_INF):
        """ log-likelihood """
        return self.clamped(string1, string2, psi, minx, miny, maxoffset) - self.logZ(string1, psi, minx, miny, maxoffset)

    def dll(self, string1, string2, psi, minx=INT_INF, miny=INT_INF, maxoffset=INT_INF):
        """ gradient of log-likelihood """
        return self.dclamped(string1, string2, psi, minx, miny, maxoffset) - self.dlogZ(string1, psi, minx, miny, maxoffset)

    def dll_is(self, string1, string2, strings_neg, weights, psi, minx=INT_INF, miny=INT_INF, maxoffset=INT_INF):
        """ importance sampled gradient """
        dpsi = self.dclamped(string1, string2, psi, minx, miny, maxoffset)
        for string, weight in zip(strings_neg, weights):
            dpsi -= weight * self.dclamped(string1, string2, psi, minx, miny, maxoffset)
        return dpsi
    
    cdef void viterbi(self, int N, int [:, :] X, double[:, :, :] g, int[:, :, :, :] bp, \
                      double[:, :, :, :] psi, int minx=INT_INF, int miny=INT_INF, int maxoffset=INT_INF):
        """ Viterbi algorithm """
        
        # assign variables
        cdef int lenP = self.lenP
        cdef int lenY = self.lenY
        cdef int[:, :] CONCAT = self.CONCAT
        cdef int[:, :] A = self.A
        cdef int[:] NEXT = self.NEXT
        cdef int[:] LENY = self.LENY
        cdef int IL = self.IL
        cdef int i, j, k, l, x, y
        cdef int p, n, z
        cdef double value
        
        # core algorithm
        for i in xrange(N + 1):
            for j in xrange(i, min(i + minx, N + 1)):
                x = X[i, j]
                if x == -1:                                                       # skip illicit input segments
                    continue
                for k in xrange(N + 1 + IL):
                    if abs(i - k) >= maxoffset:                                   # enforce constrained monotonicity
                        continue
                    for y in xrange(lenY):
                        if A[x, y] == -1:                                         # enforce alignment dictionary
                            continue
                        if LENY[y] > miny:
                            continue
                        l = k + LENY[y]
                        if l > N + IL:                                            # cannot go beyond the insertion limit
                            continue
                        if abs(j - l) >= maxoffset:                               # enforce constrained monotonicity
                            continue
                        for p in xrange(lenP):                                    # previous output Markov order
                            z = CONCAT[p, y]
                            n = NEXT[z]
                            value = psi[i, x, y, p] + g[i, k, p]
                            if value >= g[j, l, n]:                               # we found a better source
                                g[j, l, n] = value                                # dynamic program
                                bp[j, l, n, 0] = i
                                bp[j, l, n, 1] = k
                                bp[j, l, n, 2] = p
                                bp[j, l, n, 3] = y

    def decode(self, string1, psi, minx=INT_INF, miny=INT_INF, maxoffset=INT_INF):
        """ Decode the model """
        
        cdef int N = len(string1)
        cdef int IL = self.IL
        cdef int lenP = self.lenP
        X = self.extractX(string1, minx, miny, maxoffset)
        cdef double[:, :, :] g = ones((N+1, N+IL+1, lenP)) * NINF
        cdef int[:, :, :, :] bp = zeros((N+1, N+IL+1, lenP, 4), dtype=int32)
        g[0, 0, 0] = 0.0
        self.viterbi(N, X, g, bp, psi, minx, miny, maxoffset)

        # get the best value from the decode
        cdef double max_value = NINF
        cdef int kbest = -1
        cdef int pbest = -1
        for k in xrange(N+IL+1):
            for p in xrange(lenP):
                if g[N, k, p] >= max_value:
                    max_value = g[N, k, p]
                    kbest, pbest = k, p
                    
        # extract back pointers
        cdef int kcur = kbest
        cdef int pcur = pbest
        cdef int icur = N
        cdef int y, tmp1, tmp2, tmp3
        cdef vector[pair[int, int]] alignment = vector[pair[int, int]]()
        cdef vector[int] ys = vector[int]()
        alignment.push_back(pair[int, int](icur, kcur))
        while icur > 0 or kcur > 0:
            tmp1 = bp[icur, kcur, pcur, 0]
            tmp2 = bp[icur, kcur, pcur, 1]
            tmp3 = bp[icur, kcur, pcur, 2]
            y = bp[icur, kcur, pcur, 3]
            icur, kcur, pcur = tmp1, tmp2, tmp3
            alignment.push_back(pair[int, int](icur, kcur))
            ys.push_back(y)
            
        # TODO: replace push_back with something better to avoid extra O(n) computation.
        string2 = ""
        for y in list(reversed(ys)):
            string2 += self.Y.lookup(y)
        return max_value, string2, list(reversed(alignment))

    def extractX(self, string1, minx=INT_INF, miny=INT_INF, maxoffset=INT_INF):
        """ extract the X segments in the input """
        cdef int N = len(string1)
        cdef int[:, :] X = ones((N+1, N+1), dtype=int32)*-1
        cdef int i, j, x
        for i in xrange(N + 1):
            for j in xrange(i, min(i + minx, N + 1)):
                seg = string1[i:j]
                if seg in self.X:
                    x = self.X[seg]
                    X[i, j] = x
        return X

    def extractY(self, string2, minx=INT_INF, miny=INT_INF, maxoffset=INT_INF):
        """ extract the Y segments in the output """
        M = len(string2)
        Y = ones((M+1, M+1), dtype=int32)*-1
        for k in xrange(M + 1):
            for l in xrange(k, min(k + miny, M + 1)):
                seg = string2[k:l]
                if seg in self.Y:
                    y = self.Y[seg]
                    Y[k, l] = y
        return Y
