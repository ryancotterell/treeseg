from __future__ import division
from arsenal.iterview import iterview
from arsenal.alphabet import Alphabet
from termcolor import colored
from segmenter.data.data import Data, walk
from segmenter.data.data import string as to_string
from segmenter.data.data import segmentation as to_segmentation
from segmenter.data.datastructures import Tree
from segmenter.tree.tree_seg import TreeSegmenter
from segmenter.tree.tree_features import TreeFeatures
from segmenter.chunk.chunk_seg import ChunkSegmenter
from segmenter.chunk.chunk_features import ChunkFeatures
from segmenter.transducer.transducer_model import TransducerModel
from segmenter.transducer.transducer_features import TransducerFeatures
from segmenter.utils.lazyadagrad import LazyRegularizedAdagrad
import numpy as np
from numpy import zeros, zeros_like, array, logaddexp
import logging
from logging.handlers import RotatingFileHandler
import cProfile as profile
import Levenshtein
import sys
from sys import exit

# constants
# TODO: put in a util file
NINF = -np.inf
TREE = 'tree'
CHUNK = 'chunk'
ORACLE = 'oracle'
VITERBI = 'viterbi'
SAMPLE = 'sample'
LOAD = 'load'
PIPE = 'pipe'
JOINT = 'joint'
BASELINE = 'baseline'

class Segmenter(object):
    """ Segmenter """
    
    def __init__(self, train, dev, test, decode_type, split_num, log_fname=None, segmenter_type='tree', G=3, weights='weights', alphabet=None, T_L=2, T_eta=1.0, T_C=0.0000001, S_L=2, S_eta=1.0, S_C=0.00000001):
        # set up the logging system
        log = logging.getLogger('')
        log.setLevel(logging.INFO)
        format = logging.Formatter("%(asctime)s<>%(levelname)s<>%(message)s")
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(format)
        log.addHandler(ch)
        
        fh = logging.handlers.RotatingFileHandler(log_fname, maxBytes=(1048576*5), backupCount=7)
        fh.setFormatter(format)
        log.addHandler(fh)
        
        self.G = G
        self.split_num = split_num
        self.segmenter_type = segmenter_type
        self.decode_type = decode_type
        self.S_L, self.S_eta, self.S_C = S_L, S_eta, S_C
        self.T_L, self.T_eta, self.T_C = T_L, T_eta, T_C
        self.sig = "split={0},L={1},C={2},eta={3},type={4}".format(*(self.split_num, self.T_L, self.T_C, self.T_eta, self.decode_type))
        logging.info("Transducer Regularizer Type: L={0}".format(T_L))
        logging.info("Transducer Regularizer Coefficient: C={0}".format(T_C))
        logging.info("Transducer Learning Rate: eta={0}".format(T_eta))
        logging.info("Segmenter Regularizer Type: L={0}".format(S_L))
        logging.info("Segmenter Regularizer Coefficient: C={0}".format(S_C))
        logging.info("Segmenter Learning Rate: eta={0}".format(S_eta))
        
        self.weights = weights
        self.Sigma = Alphabet()
        self.Sigma.add("") # add epsilon at 0
        self.Sigma.add("o")
        self.Sigma.add("n")
        self.Sigma.add("y")
        self.Sigma.add("s")
        self.Sigma.add("e")

        if alphabet is not None:
            self.Sigma = self.Sigma.load(alphabet)
        
        # processs the data
        # TODO: modularize
        self.train = self.process(train, 100000)
        self.dev = self.process(dev, 1000)
        self.test = self.process(test, 1000)
        
        # dump the alphabet
        self.Sigma.save("alphabets/sigma-{0}.alphabet".format(self.split_num))
        # create model
        self.segmenter = None
        logging.info("Segmenter Type: {0}".format(self.segmenter_type))
        logging.info("Decoder Type: {0}".format(self.decode_type))
        
        if self.segmenter_type == TREE:
            self.segmenter = TreeSegmenter(G)
            self.features = TreeFeatures(self.G, 1000)
        elif self.segmenter_type == CHUNK:
            self.segmenter = ChunkSegmenter(G)
            self.features = ChunkFeatures(self.G, 1000)
        else:
            raise Exception('Illicit Model Type')
        
        # transducer
        self.transducer = TransducerModel(self.train, self.dev, self.test, self.Sigma, L=self.T_L, eta=self.T_eta, C=self.T_C)

        # extract features
        self.features.featurize(self.train, 'train')
        self.features.featurize(self.dev, 'dev')
        self.features.featurize(self.test, 'test')

        # dimension of data
        self.d = 2**22 + self.features.offset
        self.updater = LazyRegularizedAdagrad(self.d, L=2, C=self.S_C, eta=0.1, fudge=1e-4)
        self.updater.w[0] = 10
        self.updater.w[1] = 10

    def save_transducer(self, directory, i):
        """ save the transducer weights """
        np.save(directory+"/transducer-{0}-{1}.npy".format(*(self.sig, i)), array(self.transducer.updater.w))
        
    def save_segmenter(self, directory, i):
        np.save(directory+"/segmenter-{0}-{1}.npy".format(*(self.sig, i)), array(self.updater.w))

    def save(self, directory):
        self.save_transducer(directory, 'final')
        self.save_segmenter(directory, 'final')

    def optimize(self, t=None, load=False, transducer=None, segmenter=None, iterations=20):
        """ optimize the model """
        if load:
            assert transducer is not None
            #assert segmenter is not None
            self.load_weights(transducer, segmenter)
        if t is None:
            return
        elif t == JOINT:
            self.optimize_joint(iterations)
        elif t == PIPE:
            self.optimize_pipeline(iterations)
            
    def load_weights(self, transducer, segmenter):
        """ load weights """
        self.transducer.updater.w = np.load(transducer)
        self.updater.w = np.load(segmenter)

    def optimize_pipeline(self, iterations=10):
        """ optimize """
        
        for i in xrange(iterations):
            self.transducer.optimize(1, i)
            train_acc = self.transducer.evaluate(self.train)
            dev_acc = self.transducer.evaluate(self.dev)
            test_acc = self.transducer.evaluate(self.test)
            logging.info("transducer epoch {0} train acc: {1}".format(*(i, train_acc)))
            logging.info("transducer epoch {0} dev acc: {1}".format(*(i, dev_acc)))
            logging.info("transducer epoch {0} test acc: {1}".format(*(i, test_acc)))
            self.save_transducer(self.weights, i)

        print self.transducer.evaluate(self.dev)
        
        if self.segmenter_type == TREE:
            self.optimize_tree(iterations)
        elif self.segmenter_type == CHUNK:
            self.optimize_chunk(iterations)

    def optimize_chunk(self, iterations):
        """ optimize the model """
        for i in xrange(iterations):
            for tree in iterview(self.train, colored('Pass %s' % (i+1), 'blue')):
                gt0, gt, ge = self.features.potentials_catchup(tree, self.updater.w, self.updater)
                dgt0, dgt, dge = zeros_like(gt0), zeros_like(gt), zeros_like(ge)
                self.segmenter.dll(tree, ge, gt0, gt, dge, dgt0, dgt)
                self.features.update(tree, dge, dgt0, dgt, self.updater)
                self.updater.step += 1
                
            self.save_segmenter(self.weights, i)            
            self.decode(self.train, VITERBI)
            self.decode(self.dev, VITERBI)
            test_acc, test_f1  = self.decode(self.test, VITERBI)
            logging.info("chunk epoch {0} train acc: {1}".format(*(i, train_acc)))
            logging.info("chunk epoch {0} dev acc: {1}".format(*(i, dev_acc)))
            logging.info("chunk epoch {0} test acc: {1}".format(*(i, test_acc)))
            logging.info("chunk epoch {0} train f1: {1}".format(*(i, train_f1)))
            logging.info("chunk epoch {0} dev f1: {1}".format(*(i, dev_f1)))
            logging.info("chunk epoch {0} test f1: {1}".format(*(i, dev_f1)))

    def optimize_tree(self, iterations):
        """ optimize the model """
        for i in xrange(iterations):
            for tree in iterview(self.train, colored('Pass %s' % (i+1), 'blue')):
                psi = self.features.potentials_catchup(tree, self.updater.w, self.updater)
                dpsi = self.segmenter.dll(tree, psi)
                self.features.update(tree, dpsi, self.updater)
                self.updater.step += 1

            
            self.save_segmenter(self.weights, i)
            self.decode(self.train, VITERBI)
            self.decode(self.dev, VITERBI)
            test_acc, test_f1 = self.decode(self.test, VITERBI)
            logging.info("tree epoch {0} train acc: {1}".format(*(i, train_acc)))
            logging.info("tree epoch {0} dev acc: {1}".format(*(i, dev_acc)))
            logging.info("tree epoch {0} test acc: {1}".format(*(i, test_acc)))
            logging.info("tree epoch {0} train f1: {1}".format(*(i, train_f1)))
            logging.info("tree epoch {0} dev f1: {1}".format(*(i, dev_f1)))
            logging.info("tree epoch {0} test f1: {1}".format(*(i, test_f1)))

    def optimize_joint(self, iterations, num_samples=10, eta1=0.0, eta2=0.0):
        """ optimize jointly using importance sampling """
        # TODO: unit test
        self.updater.eta = eta1
        for i in xrange(iterations):
            samples = self.transducer.sample(self.train, num=num_samples)
            for tree, sample in iterview(zip(self.train, samples), colored('Pass %s' % (i+1), 'blue')):
                # compute approximate partition function
                logZ = NINF
                strings, weights = [], []
                for (ur, count) in sample.items():
                    score = self.transducer.ll(tree, ur)
                    if self.segmenter_type == CHUNK:
                        score += self.score_chunk(tree, ur)
                    elif self.segmenter_type == TREE:
                        score += self.score_tree(tree, ur)
                    # TODO: double check
                    logZ = logaddexp(logZ, score)
                    weights.append(score)
                    strings.append(ur)

                #TODO: make more elegant
                tmp = [] 
                for weight in weights:
                    tmp.append(weight - logZ) # TODO: double check
                weights = tmp
                
                # take a tranducer weight gradient step with the importance sampling
                self.transducer.step_is(tree, strings, weights, eta=eta2)
                # take a segmenter weight gradient step with the importance sampling
                for ur, weight in zip(sample, weights):
                    if self.segmenter_type == CHUNK:
                        self.is_chunk(tree, ur, weight)
                    elif self.segmenter_type == TREE:
                        self.is_tree(tree, ur, weight)
                self.updater.step += 1
                
    def is_chunk(self, tree, ur, weight):
        """ importance sampling gradient step tree """
        tree.update_ur(ur)
        self.features.featurize_instance(tree)
        gt0, gt, ge = self.features.potentials_catchup(tree, self.updater.w, self.updater)
        dgt0, dgt, dge = zeros_like(gt0), zeros_like(gt), zeros_like(ge)
        self.segmenter.dll(tree, ge, gt0, gt, dge, dgt0, dgt)
        dgt0 *= weight; dgt *= weight; dge *= weight
        self.features.update(tree, dge, dgt0, dgt, self.updater)

    def is_tree(self, tree, ur, weight):
        """ importance sampling gradient step chunk """
        tree.update_ur(ur)
        self.features.featurize_instance(tree)
        psi = self.features.potentials_catchup(tree, self.updater.w, self.updater)
        dpsi = self.segmenter.dll(tree, psi)
        dpsi *= weight
        self.features.update(tree, dpsi, self.updater)

    def baseline_ur(self, data):
        """ baseline ur """
        for tree in iterview(data, colored('Updating Baseline UR', 'red')):
            tree.ur_samples = []
            tree.ur_samples.append(tree.sr)
            
    def decode_ur(self, data):
        """ decodes the UR """
        for tree in iterview(data, colored('Updating Viterbi UR', 'red')):
            tree.ur_samples = []
            viterbi_ur = self.transducer.decode(tree)[1]
            tree.ur_samples.append(viterbi_ur)

    def oracle_ur(self, data):
        """ uses the oracle  UR """
        for tree in iterview(data, colored('Updating Oracle UR', 'red')):
            tree.ur_samples = []
            tree.ur_samples.append(tree.ur_gold)

    def sample_ur(self, data, num_samples=1000):
        """ samples the UR """
        samples = self.transducer.sample(data, num=num_samples)
        for tree, samples in iterview(zip(data, samples), colored('Sampling', 'red')):
            tree.ur_samples = []
            viterbi_ur = self.transducer.decode(tree)[1]
            tree.ur_samples.append(viterbi_ur)
            for sample, count in samples.items():
                tree.ur_samples.append(sample)

    def decode_chunk(self, tree, ur):
        """ decodes a chunk """
        tree.update_ur(ur)
        self.features.featurize_instance(tree)
        gt0, gt, ge = self.features.potentials_catchup(tree, self.updater.w, self.updater)
        best, segments, labels = self.segmenter.decode(tree, ge, gt0, gt)
        truth =[tree.ur[i:j] for i, j in tree.indices]
        guess = [tree.ur[i:j] for i, j in segments]
        return truth, guess

    def decode_tree(self, tree, ur):
        """ decodes a tree """
        tree.update_ur(ur)
        self.features.featurize_instance(tree)
        psi = self.features.potentials_catchup(tree, self.updater.w, self.updater)
        max_score, tree_string, max_spans = self.segmenter.argmax(tree.M, self.G, psi, tree.ur)

        gold_spans = set(tree.spans)
        guess_spans = set(tree.spans)
        p, r = 0.0, 0.0
        for span in gold_spans:
            if span in guess_spans:
                p += 1.0
        p /= len(gold_spans)
        for span in guess_spans:
            if span in gold_spans:
                r += 1.0
        r /= len(guess_spans)
        f1 = (2*p*r)/(p+r)

        # TODO: horrible hack
        segmentation = tree_string.replace("(", "").replace(")", "").replace(" ", "")
        for i in xrange(100):
            segmentation = segmentation.replace(str(i), "")
        segmentation = segmentation.split(":")
        guess = segmentation[:-1]
        truth = [x[0] for x in to_segmentation(tree.root)]
        return truth, guess, f1

    def score_chunk(self, tree, ur):
        """ scores a chunk """
        tree.update_ur(ur)
        self.features.featurize_instance(tree)
        M = tree.M
        gt0, gt, ge = self.features.potentials_catchup(tree, self.updater.w, self.updater)
        return self.segmenter.logZ(ge, gt0, gt, M)

    def score_tree(self, tree, ur):
        """ scores a tree """
        tree.update_ur(ur)
        self.features.featurize_instance(tree)
        M = tree.M
        psi = self.features.potentials_catchup(tree, self.updater.w, self.updater)
        return self.segmenter.logZ(M, self.G, psi)
        
    def decode(self, data, data_type, decode_type=None, sep=u"#"):
        """ decode the chunker """
        if decode_type is None:
            decode_type = self.decode_type
        
        if decode_type == ORACLE:
            self.oracle_ur(data)
        elif decode_type == BASELINE:
            self.baseline_ur(data)
        elif decode_type == VITERBI:
            self.decode_ur(data)
        elif decode_type == SAMPLE:
            self.sample_ur(data)
        else:
            raise Exception('Illicit Decode Type')

        ur_correct, ur_total = 0, 0
        correct, f1, tree_f1, lev, total = 0, 0, 0, 0, 0
        for tree in iterview(data, colored('Decoding', 'red')):
            max_ur, max_score = None, NINF
            counter = 0
            for ur in tree.ur_samples:
                tree.update_ur(ur)
                counter += 1
                score = 0.0
                score = self.transducer.ll(tree, ur)
                #print 
                #print "LL", self.transducer.ll(tree, ur)
                if self.segmenter_type == CHUNK:
                    score += self.score_chunk(tree, ur)
                    #print "SCORE", self.score_chunk(tree, ur)
                    #print ur
                    #raw_input()
                elif self.segmenter_type == TREE:
                    score += self.score_tree(tree, ur)
                # take the best importance sample
                if score >= max_score:
                    max_score = score
                    max_ur = ur
                    #print "counter", counter
            if max_ur == tree.ur_gold:
                ur_correct += 1
            ur_total += 1
            truth, guess, tree_f1_tmp = None, None, None
            if self.segmenter_type == CHUNK:
                truth, guess = self.decode_chunk(tree, max_ur)
            elif self.segmenter_type == TREE:
                truth, guess, tree_f1_tmp = self.decode_tree(tree, max_ur)
                tree_f1 += tree_f1_tmp
                    
            # ACCURACY    
            if truth == guess:
                correct += 1
            # LEVENSHTEIN
            lev += Levenshtein.distance(sep.join(truth), sep.join(guess))
            # F1
            set1, set2 = set(guess), set(truth)
            p, r = 0, 0
            for e in set1:
                if e in set2:
                    p += 1
            for e in set2:
                if e in set1:
                    r += 1
            p /= len(set1)
            r /= len(set2)
            if p+r > 0:
                f1 += 2*p*r/(p+r)
            total += 1

        logging.info("decoder type: {0}".format(decode_type))
        logging.info("{0} ur acc: {1}".format(*(data_type, ur_correct/total)))
        logging.info("{0} seg acc: {1}".format(*(data_type, correct/total)))
        logging.info("{0} f1: {1}".format(*(data_type, f1/total)))
        logging.info("{0} edit: {1}".format(*(data_type, lev/total)))
        if self.segmenter_type == TREE:
            logging.info("{0} tree f1: {1}".format(*(data_type, tree_f1/total)))
           
    def process(self, fname, maximum=100):
        """ 
        Put the string data into the data structures
        necessary for training and decoding the model. 
        """
        processed = []
        data = Data(fname)
        for counter, (sr, (tree, (indices, index_labels))) in enumerate(data):
            if counter == maximum:
                break
            ur = to_string(tree)
            for s in list(sr):
                self.Sigma.add(s)
            for s in list(ur):
                self.Sigma.add(s)
                
            spans, labels = [], []
            for node in walk(tree):
                spans.append((node.i, node.j))
                labels.append(node.label)
            t = Tree(self.G, sr, ur, spans, labels, indices, index_labels, len(spans), tree)
            processed.append(t)

        return processed

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--train', type=str)
    p.add_argument('--dev', type=str)
    p.add_argument('--test', type=str)
    p.add_argument('--type', type=str)
    p.add_argument('--decode', choices=('oracle', 'viterbi', 'sample', 'baseline', 'all'))
    p.add_argument('--log', type=str)
    p.add_argument('--split', type=str)
    p.add_argument('--T_C', type=float)
    p.add_argument('--S_C', type=float)
    p.add_argument('--S_L', type=float)
    p.add_argument('--S_eta', type=float)
    args = p.parse_args()

    #f_alphabet = "alphabets/sigma-1.alphabet"
    #f_transducer = "weights/transducer-split={0},L=2,C=1e-10,eta=1.0,type=viterbi-6.npy".format(args.split)
    #f_segmenter = "weights/segmenter-split=1,L=2,C=1e-09,eta=1.0,type=viterbi-5.npy"
    f_alphabet, f_transducer, f_segmenter = None, None, None
    

    segmenter = Segmenter(args.train, args.dev, args.test, args.decode, args.split, log_fname=args.log, segmenter_type=args.type, T_C=args.T_C, alphabet=f_alphabet, S_C=args.S_C)
    #segmenter.optimize(t=PIPE, load=False, transducer=f_transducer)#, segmenter=f_segmenter)

    #egmenter = Segmenter(args.train, args.dev, args.test, args.decode, args.split, log_fname=args.log, segmenter_type=args.type, T_C=args.T_C, alphabet=f_alphabet)
    #segmenter.optimize(t=PIPE, load=True, transducer=f_transducer, segmenter=f_segmenter)
    #segmenter.optimize(t=PIPE)

    if args.decode == 'all':
        for t in [VITERBI, ORACLE, BASELINE, SAMPLE]:
            segmenter.decode_type = t
            segmenter.decode(segmenter.dev, 'dev')
            segmenter.decode(segmenter.test, 'test')

    else:
        segmenter.decode(segmenter.dev, 'dev')
        
    #segmenter.optimize_joint(2)
    #print segmenter.decode(segmenter.dev)

    #segmenter.save("weights")
    
    #profile.runctx("segmenter.optimize(10)", locals(), globals())
    #segmenter.decode(segmenter.dev)
    
    
