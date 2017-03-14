from segmenter.utils.sparse cimport SparseBinaryVector
from segmenter.utils.lazyadagrad cimport LazyRegularizedAdagrad
from arsenal.iterview import iterview
from arsenal.timer import timeit
from arsenal.alphabet import Alphabet
from termcolor import colored
import numpy as np
from numpy import empty, zeros
import enchant

from libc.math cimport exp, log, log1p, sqrt
from libc.float cimport DBL_MIN, DBL_MAX

PPAD = "^"
SPAD = "$"

cdef inline double sign(double x) nogil:
    return 1 if x >= 0 else -1
        
cdef class ChunkFeatures(object):
    """ Feature hashing for the chunk features """

    # number of labels
    cdef int G
    cdef public int offset
    cdef object words
    cdef object d
    
    def __init__(self, G, offset):
        self.G = G
        self.offset = offset
        self.words = Alphabet()
        self.d = enchant.Dict("en_US")
        #self.d = {}
        
    def featurize(self, data, msg, prefix_context_size=5, suffix_context_size=5):
        """ Maps a word, segmentation pair to a instance """
        for instance in iterview(data, colored('Features (%s)' % msg, 'green')):
            self.featurize_instance(instance, prefix_context_size, suffix_context_size)

    def featurize_instance(self, instance, prefix_context_size=5, suffix_context_size=5):
        """ featurize a single instance """
        M = instance.M
        ur = instance.ur
        segments = empty((M+1, M+1), dtype='object')
        for i in xrange(1, M+1):
            prefix_context = []
            for k in xrange(1, prefix_context_size):
                if i-k < 0:
                    break
                prefix = (PPAD+ur+SPAD)[max(0, i-k):i]
                prefix_context.append(prefix)
                        
            for j in xrange(i+1, M+2):
                suffix_context = []
                for k in xrange(1, suffix_context_size+1):
                    if k > M+2:
                        break
                    suffix = (PPAD+ur+SPAD)[j:min(M+2,j+k)]
                    suffix_context.append(suffix)
                seg = ur[i-1:j-1]
                # features
                features = [abs(hash(x)) for x in [
                    ('seg', seg)
                ] + prefix_context + suffix_context]

                if self.d.check(seg) and len(seg) > 3:
                    features.append(0)
                    
                segments[i-1, j-1] = SparseBinaryVector(features)
                
        instance.features = segments
    
    cpdef void update(self, object x, double[:, :, :] d_ge, double[:] d_gt0, double[:, :] d_gt, LazyRegularizedAdagrad updater):
         """ computes the potentials """

         cdef SparseBinaryVector f
         cdef int y, yp, t, tp, k, M
         cdef double v
         M = x.M
         for tp in xrange(M+1):
             for t in xrange(tp+1, M+1):
                 f = x.features[tp, t]

                 for y in xrange(self.G):
                     if tp == 0:
                         v = d_gt0[y]
                         k = y
                         updater.update_active(k, v)
                     else:
                         for yp in xrange(self.G):
                            # transition feature
                            v = d_gt[yp, y]
                            # single tag feature
                            k = y
                            updater.update_active(k, v)
                            # conjunction of tag features
                            k = (yp+1)*self.G + y
                            updater.update_active(k, v)

                     v = d_ge[y, tp, t]
                     for i in range(f.length):
                         # conjunction with just the label
                         k = f.keys[i]*self.G + y
                         k &= 0b1111111111111111111111   # mod 2**22
                         k = k + self.offset
                         updater.update_active(k, v)

    cpdef object potentials_catchup(self, object x, double[:] w, LazyRegularizedAdagrad updater):
         """ Compute log-potentials and lazily update `w` using L1/L2-regularized AdaGrad
          catch updates described in
          http://nlp.cs.berkeley.edu/pubs/Kummerfeld-BergKirkpatrick-Klein_2015_Learning_paper.pdf

         """
         cdef SparseBinaryVector f
         cdef int t, tp, y, yp, i, s, k, M
         cdef double z, zy
         M = x.M
         
         cdef double[:] gt0 = zeros((self.G))
         cdef double[:, :] gt = zeros((self.G, self.G))
         cdef double[:, :, :] ge = zeros((self.G, M+1, M+1))
         
         for tp in xrange(M+1):
             for t in xrange(tp+1, M+1):
                 f = x.features[tp, t]
                 for y in xrange(self.G):
                     if tp == 0:
                         k = y
                         gt0[y] += updater.catchup(k)
                     else:
                         for yp in xrange(self.G):
                             # single tag
                             k = y
                             gt[yp, y] += updater.catchup(k)
                             k = (yp+1)*self.G + y
                             gt[yp, y] += updater.catchup(k)
                             
                     for i in xrange(f.length):
                         # conjunction with just the label
                         k = f.keys[i]*self.G + y
                         k &= 0b1111111111111111111111 # mod 2**22
                         k = k + self.offset
                         ge[y, tp, t] += updater.catchup(k)

         return gt0, gt, ge
