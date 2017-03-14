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

from numpy import empty

from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef class Tree(object):
    """ Tree object """

    cdef readonly int N
    cdef public int M
    cdef readonly int G
    cdef readonly int size
    cdef public object ur
    cdef readonly object ur_gold
    cdef readonly object sr
    cdef readonly vector[pair[int, int]] spans
    cdef readonly vector[int] labels
    cdef readonly vector[pair[int, int]] indices
    cdef readonly vector[int] index_labels
    cdef public object[:, :] features
    cdef public object[:, :] input_features
    cdef public object root
    cdef public list ur_samples
