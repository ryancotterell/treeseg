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

from arsenal.iterview import iterview
from arsenal.timer import timeit
from arsenal.alphabet import Alphabet
from termcolor import colored
import numpy as np
from numpy import empty, zeros
import enchant

from libc.math cimport exp, log, log1p, sqrt
from libc.float cimport DBL_MIN, DBL_MAX
from segmenter.utils.sparse cimport SparseBinaryVector
from segmenter.utils.lazyadagrad cimport LazyRegularizedAdagrad

cdef object PPAD = "^"
cdef object SPAD = "$"

cdef inline double sign(double x) nogil:
    return 1 if x >= 0 else -1

cdef class TreeFeatures(object):
    """ Feature hashing for the segment features """

    # number of labels
    cdef public int G, offset
    cdef public object words, segments, feature_hash, d
    
    def __init__(self, G, offset):
        self.G = G
        self.offset = offset
        self.words = Alphabet()
        self.segments = Alphabet()
        self.feature_hash = Alphabet() # perfect feature hash
        self.d = enchant.Dict("en_US")

    def featurize(self, data, msg, prefix_context_size=4, suffix_context_size=5):
        for tree in iterview(data, colored('Features (%s)' % msg, 'green')):
            self.featurize_instance(tree, prefix_context_size, suffix_context_size)
    
    def featurize_instance(self, tree, prefix_context_size=5, suffix_context_size=5):
        """ Maps a word, segmentation pair to a instance """
        cdef int G = self.G
        M = tree.M
        ur = tree.ur
        features = empty((M+1, M+1), dtype='object')
        for i in xrange(1, M+2):
            prefix_context = []
            for k in xrange(1, prefix_context_size):
                if i-k < 0:
                    break
                prefix = (PPAD+ur+SPAD)[max(0, i-k):i]
                prefix_context.append(prefix)
            for j in xrange(i+1, M+2):
                suffix_context = []
                for k in xrange(1, suffix_context_size+1):
                    if j+k > M+2:
                        break
                    suffix = (PPAD+ur+SPAD)[j:min(M+2,j+k)]
                    suffix_context.append(suffix)
                seg = ur[i-1:j-1]
                # features
                pre = [self.feature_hash[('p', x)]+self.offset for x in prefix_context]
                suf = [self.feature_hash[('s', x)]+self.offset for x in suffix_context]
                
                self.segments.add(seg)
                hashed_features = [self.segments[('seg', seg)]+self.offset] + pre + suf
                
                if self.d.check(seg) and len(seg) > 3:
                    hashed_features.append(1)
                features[i-1, j-1] = SparseBinaryVector(hashed_features)
        tree.features = features
    
    cpdef void update(self, object tree, double[:, :, :] dpsi, LazyRegularizedAdagrad updater):
         """ computes the potentials """

         cdef SparseBinaryVector f
         cdef int M, G, i, j, k, l, g
         cdef double v, vg
         M, G = tree.M, self.G

         for i in xrange(M+1):
             for j in xrange(i+1, M+1):
                 f = tree.features[i, j]
                 vg = 0.0
                 for g in xrange(G):
                     v = dpsi[i, j, g]
                     vg += v
                     for l in xrange(f.length):
                         k = f.keys[l]*(G+1) + g
                         k &= 0b1111111111111111111111   # mod 2**22
                         updater.update_active(k, v)
                 #for l in range(f.length):
                 #    k = f.keys[l]
                 #    k &= 0b1111111111111111111111   # mod 2**22
                 #    updater.update_active(k, vg)
                     
    cpdef object potentials_catchup(self, object tree, double[:] w, LazyRegularizedAdagrad updater):
         """ Compute log-potentials and lazily update `w` using L1/L2-regularized AdaGrad
          catch updates described in
          http://nlp.cs.berkeley.edu/pubs/Kummerfeld-BergKirkpatrick-Klein_2015_Learning_paper.pdf
         """
         cdef SparseBinaryVector f
         cdef int i, j, g, t, tp, y, yp, s, k, l, M
         cdef double z, zy
         M, G = tree.M, self.G
         
         cdef double[:, :, :] psi = zeros((M+1, M+1, G))

         for i in xrange(M+1):
             for j in xrange(i+1, M+1):
                 f = tree.features[i, j]
                 vg = 0.0
                 #for l in range(f.length):
                 #    k = f.keys[l]
                 #    k &= 0b1111111111111111111111   # mod 2**22
                 #    vg += updater.catchup(k)
                 for g in xrange(G):
                     for l in xrange(f.length):
                         k = f.keys[l]*(G+1) + g
                         k &= 0b1111111111111111111111   # mod 2**22
                         psi[i, j, g] += updater.catchup(k) + vg
         return psi
     
