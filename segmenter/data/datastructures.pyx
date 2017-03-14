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

cdef class Tree(object):
    """ tree object """
    
    def __init__(self, G, sr, ur, spans, labels, indices, index_labels, size, root):
        self.N = len(sr)
        self.M = len(ur)
        self.G = G
        self.sr = sr
        self.ur = ur
        self.ur_gold = ur
        self.spans = spans
        self.labels = labels
        self.root = root
        self.indices = indices
        self.index_labels = index_labels
        self.ur_samples = []
        # redundant, but ensure
        # cythonization
        self.size = size
        assert len(self.spans) == size
        assert len(self.labels) == size

    def update_ur(self, ur):
        """ updates the ur """
        self.M = len(ur)
        self.ur = ur
