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

from segmenter.utils.sparse cimport SparseBinaryVector
from segmenter.utils.lazyadagrad cimport LazyRegularizedAdagrad

from arsenal.iterview import iterview
from arsenal.timer import timeit
from arsenal.alphabet import Alphabet
from termcolor import colored
import numpy as np
from numpy import empty, zeros

from libc.math cimport exp, log, log1p, sqrt
from libc.float cimport DBL_MIN, DBL_MAX

PPAD = "^"
SPAD = "$"

cdef inline double sign(double x) nogil:
    return 1 if x >= 0 else -1
        
cdef class TransducerFeatures(object):
    """ Feature hashing for the transducer features """

    # number of labels
    cdef public int offset
    cdef int lenX, lenY, lenP, lenA
    cdef public int[:, :] A
    cdef int[:, :] COPY
    cdef public object words, d, X, Y, P
    cdef object[:] features_P, features_X, features_y
    
    def __init__(self, X, Y, P):
        self.X, self.Y, self.P = X, Y, P

        # make sure we cannot add any new elements
        # will throw an error if we do
        self.X.freeze(); self.Y.freeze(); self.P.freeze()
        # set lengths
        self.lenX, self.lenY, self.lenP = len(self.X), len(self.Y), len(self.P)

        # alignment alphabet
        self.A = zeros((self.lenX, self.lenY), dtype=np.int32)
        self.d = {}
        #self.features_P, self.features_X, self.features_Y = None, None, None

        # make the alignment alphabet
        self.make_alignment_alphabet()
        self.featurize_P()
        
    def make_alignment_alphabet(self):
        """ makes the alignment alphabet """
        self.COPY = zeros((self.lenX, self.lenY), dtype=np.int32)
        cdef int counter = 2 # starts at one because of copy
        cdef int xi, yi
        for x, xi in self.X.items():
            for y, yi in self.Y.items():
                self.A[xi, yi] = counter
                if x == y:
                    self.COPY[xi, yi] = 1
                counter += 1
        self.lenA = counter
        
    def featurize_P(self):
        """ extract features on P """
        self.features_P = empty((len(self.P)), dtype='object')
        cdef int pi, k
        for p, pi in self.P.items():
            features = []
            for k in xrange(len(p)+1):
                suffix = p[k:]
                suffixi = self.P[suffix] + self.lenA

                # get the maximum offset value
                self.offset = max(self.offset, self.P[suffix] + self.lenA)
                
                features.append(suffixi)
            #self.features_P[pi] = SparseBinaryVector(features)
            self.features_P[pi] = SparseBinaryVector([])
        self.offset += 1

    def featurize_X(self):
        """ Extract features on X """
        self.features_X = empty((len(self.X)), dtype='object')
        cdef int xi, k
        for x, xi in self.X.items():
            features = []
            for k in xrange(len(x)+1):
                suffix = x[k:]
                features.append(('x', suffix))
            self.features_X[xi] = SparseBinaryVector([abs(hash(x)) for x in features])

    def featurize_Y(self):
        """ Extract features on Y """
        self.features_Y = empty((len(self.Y)), dtype='object')
        cdef int yi, k
        for y, yi in self.Y.items():
            features = []
            for k in xrange(len(y)+1):
                suffix = y[k:]
                features.append(('y', suffix))
            self.features_Y[yi] = SparseBinaryVector([abs(hash(y)) for y in features])
        
    def featurize(self, data, msg, prefix_context_size=6, suffix_context_size=6):
        """ Maps a word  pair to a instance """

        for instance in iterview(data, colored('Features (%s)' % msg, 'green')):
            N = instance.N   # TODO: fix reversed
            sr = instance.sr
            input_features = empty((N+1, len(self.X)), dtype='object')
            for i in xrange(1, N+2):
                for xi in xrange(len(self.X)):
                    input_features[i-1, xi] = SparseBinaryVector([])
                prefix_context = []
                for k in xrange(1, prefix_context_size):
                #for k in xrange(prefix_context_size):
                    if i-k < 0:
                        break
                    prefix = (PPAD+sr+SPAD)[max(0, i-k):i]
                    prefix_context.append(prefix)
                for j in xrange(i, N+3):
                    x = sr[i-1:j-1]
                    if x not in self.X:
                        continue
                    xi = self.X[x]
                    suffix_context = []
                    for k in xrange(1, suffix_context_size+1):
                        if k > N+2:
                            break
                        suffix = (PPAD+sr+SPAD)[j:min(N+2,j+k)]
                        suffix_context.append(suffix)
                        
                    # features
                    #features = [abs(hash(z)) for z in [
                    #    ('x', x)
                    #] + prefix_context + suffix_context]
                    
                    pre = [('prefix', x) for x in prefix_context]
                    suf = [('suffix', x) for x in suffix_context]
                    other = [('other', "iert" in sr)]

                    features = [abs(hash(z)) for z in [('x', x)] + pre + suf + other]
                    input_features[i-1, xi] = SparseBinaryVector(features)

                    # for x, y in zip(pref, features):
                    #     print x, y
                    #     print features
                    # for tt in xrange(input_features[i-1, xi].length):
                    #     print input_features[i-1, xi].keys[tt]
                    # raw_input()
                    
            instance.input_features = input_features

    cpdef void update(self, object instance, double[:, :, :, :] dpsi, LazyRegularizedAdagrad updater):
         """ computes the potentials """

         cdef SparseBinaryVector f, fp
         cdef int t, i, xi, yi, pi, ai, copy, feat
         cdef int[:, :] A = self.A
         cdef int[:, :] COPY = self.COPY
         cdef double v, vp
         cdef int N = instance.N
         for t in xrange(N+1):
             for xi in xrange(self.lenX):
                f = instance.input_features[t, xi]
                for yi in xrange(self.lenY):
                    ai = A[xi, yi]
                    copy = COPY[xi, yi]
                    vp = 0.0
                    for pi in xrange(self.lenP):
                        fp = self.features_P[pi]

                        # TODO: check sign
                        v = -dpsi[t, xi, yi, pi]
                        vp += v

                        feat = (pi+1) * self.lenA + ai
                        updater.update_active(feat, v)
                        
                        for i in xrange(fp.length):
                            # NO HASHING
                            feat = fp.keys[i] + self.lenA
                            #assert feat < self.offset
                            updater.update_active(feat, v)
                            
                            # HASHING
                            feat = fp.keys[i] * self.lenA + ai
                            feat &= 0b1111111111111111111111   # mod 2**22
                            feat = feat + self.offset
                            updater.update_active(feat, v)
                                 
                            if copy == 1:
                                feat = fp.keys[i] * self.lenA
                                feat &= 0b1111111111111111111111   # mod 2**22
                                feat = feat + self.offset
                                updater.update_active(feat, v)


                        for i in xrange(f.length):
                             feat = f.keys[i] * pi * self.lenA + ai + pi
                             feat &= 0b1111111111111111111111   # mod 2**22
                             feat = feat + self.offset
                             updater.update_active(feat, v)

                    for i in xrange(f.length):
                         feat = f.keys[i] * self.lenA + ai
                         feat &= 0b1111111111111111111111   # mod 2**22
                         feat = feat + self.offset
                         updater.update_active(feat, vp)

                         if copy == 1:
                             feat = f.keys[i] * self.lenA
                             feat &= 0b1111111111111111111111   # mod 2**22
                             feat = feat + self.offset
                             updater.update_active(feat, vp)
                    
                    # UPDATE HERE -- no hashing
                    updater.update_active(ai, vp)
                    if copy == 1:
                        updater.update_active(0, vp)
                    else:
                        updater.update_active(1, vp)
                        
    cpdef object potentials_catchup(self, object instance, double[:] w, LazyRegularizedAdagrad updater):
         """ Compute log-potentials and lazily update `w` using L1/L2-regularized AdaGrad
          catch updates described in
          http://nlp.cs.berkeley.edu/pubs/Kummerfeld-BergKirkpatrick-Klein_2015_Learning_paper.pdf
         """
         
         cdef SparseBinaryVector f, fp
         cdef int t, i, xi, yi, pi, ai, copy, feat
         cdef double z, zy
         cdef int N = instance.N
         
         cdef double[:, :, :, :] psi = zeros((N+1, self.lenX, self.lenY, self.lenP))
         cdef int[:, :] A = self.A
         cdef int[:, :] COPY = self.COPY

         for t in xrange(N+1):
             for xi in xrange(self.lenX):
                # TODO: check indices
                f = instance.input_features[t, xi]
                for yi in xrange(self.lenY):
                    ai = A[xi, yi]
                    copy = COPY[xi, yi]
                    zy = 0.0

                    zy += updater.catchup(ai)
                    if copy == 1:
                        zy += updater.catchup(0)
                    else:
                        zy += updater.catchup(1)

                    for i in xrange(f.length):
                         feat = f.keys[i] * self.lenA + ai
                         feat &= 0b1111111111111111111111   # mod 2**22
                         feat = feat + self.offset
                         zy += updater.catchup(feat)

                         if copy == 1:
                             feat = f.keys[i] * self.lenA
                             feat &= 0b1111111111111111111111   # mod 2**22
                             feat = feat + self.offset
                             zy += updater.catchup(feat)
                            
                    for pi in xrange(self.lenP):
                        fp = self.features_P[pi]
                        z = 0.0

                        feat = (pi+1) * self.lenA + ai
                        z += updater.catchup(feat)
                        
                        for i in xrange(fp.length):
                            # NO HASHING
                            feat = fp.keys[i] + self.lenA
                            z += updater.catchup(feat)
                            
                            # HASHING
                            feat = fp.keys[i] * self.lenA + ai
                            feat &= 0b1111111111111111111111   # mod 2**22
                            feat = feat + self.offset
                            z += updater.catchup(feat)
                                 
                            if copy == 1:
                                feat = fp.keys[i] * self.lenA
                                feat &= 0b1111111111111111111111   # mod 2**22
                                feat = feat + self.offset
                                z += updater.catchup(feat)

                        for i in xrange(f.length):
                            feat = f.keys[i] * self.lenA * pi + ai + pi
                            feat &= 0b1111111111111111111111   # mod 2**22
                            feat = feat + self.offset
                            zy += updater.catchup(feat)
                            
                        # TODO: double check indices
                        psi[t, xi, yi, pi] = zy + z
         return psi
