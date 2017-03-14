from __future__ import division
from segmenter.utils.lazyadagrad import LazyRegularizedAdagrad
from segmenter.transducer.transducer_features import TransducerFeatures
from segmenter.transducer.transducer import Transducer
from arsenal.alphabet import Alphabet
from arsenal.iterview import iterview
from termcolor import colored
import numpy as np
from numpy import array, zeros, zeros_like
import cProfile as profile

MINX=6
MINY=6

class TransducerModel(object):
    """ Transducer model """

    def __init__(self, train, dev, test, Sigma, IL=6, L=2, eta=0.01, C=0.0001):
        self.train = train
        self.dev = dev
        self.test = test
        self.Sigma = Sigma
        assert self.Sigma[""] == 0
        self.IL = IL
        self.C = C
        self.L = L
        self.eta = eta

        # X and Y
        self.X, self.Y = Alphabet(), Alphabet()
        self.X.add(""); self.Y.add("")
        for s, si in self.Sigma.items():
            if si == 0:
                continue
            self.X.add(s)
        for s, si in self.Sigma.items():
            if si == 0:
                continue
            self.Y.add(s)
        self.X.freeze(); self.Y.freeze()

        # first order (possibly extend)
        self.P = Alphabet()
        self.P.add("")
        for s, si in self.Sigma.items():
            if si == 0:
                continue
            self.P.add(s)
        self.P.add("oo")
        self.P.add("nn")
        self.P.add("yy")
        self.P.add("ss")
        self.P.add("ee")
        self.P.freeze()
        
        # create Z
        self.Z = Alphabet()
        self.Z[""] = 0
        for p, pi in self.P.items():
            for o, oi in self.Y.items():
                 z = p+o
                 self.Z.add(z)
        self.Z.freeze()
        
        # model
        self.model = Transducer(self.Sigma, self.X, self.Y, self.P, self.Z, IL = self.IL)
        self.features = TransducerFeatures(self.X, self.Y, self.P)
        self.features.featurize(self.train, 'train')
        self.features.featurize(self.dev, 'dev')
        self.features.featurize(self.test, 'test')
        
        self.d = 2**22 + self.features.offset
        self.updater = LazyRegularizedAdagrad(self.d, L=self.L, C=self.C, eta=self.eta, fudge=1e-4)
        self.updater.w[0] = 10.0
        self.updater.w[1] = -10.0

    def optimize(self, iterations=10, start=0):
        """ optimize the model  """
        #np.random.shuffle(self.train)
        for i in xrange(iterations):
            for instance in iterview(self.train, colored('Pass %s' % (i+1+start), 'blue')):
                psi = self.features.potentials_catchup(instance, self.updater.w, self.updater)
                dpsi = zeros_like(psi)
                x, y = instance.sr, instance.ur
                #print "LL", self.model.ll(x, y, psi, minx=MINX, miny=MINY)
                dpsi = self.model.dll(x, y, psi, minx=MINX, miny=MINY)
                self.features.update(instance, dpsi, self.updater)
                self.updater.step += 1

    def step_is(self, tree, strings, weights, eta=0.0):
        """ optimize the model  """
        self.updater.eta = eta
        psi = self.features.potentials_catchup(tree, self.updater.w, self.updater)
        dpsi = zeros_like(psi)
        dpsi = self.model.dll_is(tree.sr, tree.ur, strings, weights, psi, minx=MINX, miny=MINY)
        self.features.update(tree, dpsi, self.updater)
        self.updater.step += 1
                
    def sample(self, data, num=1000):
        """ sample """
        samples = []
        inside = 0
        correct1, correct2, total = 0, 0, 0
        for instance in iterview(data, colored('Sampling', 'green')):
            psi = self.features.potentials_catchup(instance, self.updater.w, self.updater)
            sr = instance.sr
            dist = {}
            for s in self.model.sample(sr, psi, num=num):
                output = ""
                for x, y in s:
                    output += y
                if output not in dist:
                    dist[output] = 0
                dist[output] += 1

            count = dist[instance.ur_gold] if instance.ur_gold in dist else 0
            decoded = self.decode(instance)[1]

            if decoded != instance.ur_gold and count > 0:
                inside += 1
            if decoded == instance.ur_gold:
                correct1 += 1
            if instance.ur_gold in dist:
                correct2 += 1
            total += 1
            samples.append(dist)
            
        # TODO: put into log
        #print ; print inside
        #print correct1 / total, correct2 / total
        return samples

    def decode(self, instance):
        """ Decodes an instance """
        psi = self.features.potentials_catchup(instance, self.updater.w, self.updater)
        ur1 = instance.ur
        results = self.model.decode(instance.sr, psi, minx=MINX, miny=MINY)
        return results
        
    def evaluate(self, data, maximum=100000000):
        """ decode the model """
        correct, total = 0, 0
        counter = 0
        for instance in iterview(data, colored('Decoding', 'red')):
            if counter == maximum:
                break
            psi = self.features.potentials_catchup(instance, self.updater.w, self.updater)
            ur1 = instance.ur
            results = self.model.decode(instance.sr, psi, minx=MINX, miny=MINY)
            ll = self.model.ll(instance.sr, ur1, psi, minx=MINX, miny=MINY)
            score, ur2 = results[0], results[1]
            if ur1 == ur2:
                correct += 1
            print ur1, ur2
            total += 1
            counter += 1
        print
        return float(correct) / total

    def ll(self, tree, ur):
        """ gets the log-likelihood """
        psi = self.features.potentials_catchup(tree, self.updater.w, self.updater)
        return self.model.ll(tree.sr, ur, psi, minx=MINX, miny=MINY)
    
if __name__ == "__main__":
    data = [("hablar", "hablando"), ("comer", "comiendo")]
    Sigma = Alphabet()
    Sigma.add("")
    for (x, y) in data:
        for c in list(x):
            Sigma.add(c)
        for c in list(y):
            Sigma.add(c)
            
    tm = TransductionModel(Sigma, data)
    profile.runctx("tm.train()", locals(), globals())
