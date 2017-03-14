from segmenter.tree.tree_seg import TreeSegmenter
from segmenter.segmenter import Segmenter
from segmenter.tree.enumerate_trees import Enumerator
import numpy as np
from numpy import zeros, ones, zeros_like, ones_like, log, exp, allclose, array

NINF = -np.inf

class TestTreeSegmenter(Segmenter):

    def test_enumerate_logZ(self, string):
        """ enumeration test of the log partition function """
        N = len(string)
        G = self.G
        psi = np.random.rand(N+1, N+1, G)
        #psi = zeros((N+1, N+1, G))
        te = Enumerator()
        for seg in sorted(te.enumerate_segmentations(string), key=lambda x: len(x)):
            for lab in te.enumerate_labelings(seg, G):
                te.go(lab)
        # enumeration of the partition function
        Z = 0.0
        for root in te.trees:
            score = 1.0
            for (i, j, l) in te.walk(root, string):
                score *= exp(psi[i, j, l])
            Z += score
        logZ = log(Z)

        assert allclose(logZ, self.logZ(N, G, psi), atol=0.01)
        grad = array(self.dlogZ(N, G, psi))

    def test_enumerate_max(self, string):
        """ enumeration test of Viterbi and decoding """
        N = len(string)
        G = self.G
        psi = log(np.random.rand(N+1, N+1, G))
        te = Enumerator()
        for seg in sorted(te.enumerate_segmentations(string), key=len):
            for lab in te.enumerate_labelings(seg, G):
                te.go(lab)
        # enumeration of the partition function
        max_score, max_tree = NINF, None
        for root in te.trees:
            score = 0.0
            for (i, j, l) in te.walk(root, string):
                score += psi[i, j, l]
            if score >= max_score:
                max_score = score
                max_tree = root

        max_score2, print2 = t.decode(N, G, psi, string)
        print1 = te.display(max_tree)
        assert print1 == print2
        assert allclose(max_score, max_score2, atol=0.01)
        
    def test_grad(self):
        """ checks the gradient """
        tree = self.train[0]
        N, G = tree.N, self.G
        psi = np.random.rand(N+1, N+1, G)
        dpsi_fd = self.finite_difference(tree, psi)
        assert allclose(dpsi_fd, self.dll(tree, psi), atol=0.001)

    def test_overfit(self, iterations=100):
        """ tests whether we can overfit the toy data """
        tree = self.train[0]

        N, G = tree.N, self.G
        psi = np.random.rand(N+1, N+1, G)
        psi = zeros_like(psi)
        for i in xrange(iterations):
            print self.ll(tree, psi)
            psi -= self.dll(tree, psi)
            
    def finite_difference(self, tree, psi, eps=0.001):
        """ finite difference check """
        N, G = tree.N, self.G
        dpsi_fd = zeros_like(psi)
        for i in xrange(N+1):
            for j in xrange(N+1):
                for n in xrange(G):
                    psi[i, j, n] += eps
                    val1 = self.ll(tree, psi)
                    psi[i, j, n] -= 2*eps
                    val2 = self.ll(tree, psi)
                    psi[i, j, n] += eps
                    dpsi_fd[i, j, n] = (val1-val2)/(2*eps)
        return dpsi_fd
        
if __name__ == "__main__":
    t = TestTreeSegmenter("data/test", None, None)
    #t.test_grad()
    #string = "abcd"
    #t.test_enumerate_logZ(string)
    #t.test_enumerate_max(string)
    t.test_overfit()
