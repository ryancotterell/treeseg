import itertools as it

class Node(object):
    pass

class Terminal(Node):
    def __init__(self, string, i, j, label):
        self.string = string
        self.i, self.j = i, j
        self.label = label

    def __str__(self):
        return str(self.string) + ":" + str(self.label)
        
class NonTerminal(Node):
    def __init__(self, node1, node2, i, j, label):
        self.node1, self.node2 = node1, node2
        self.i, self.j = i, j
        self.label = label

class Enumerator(object):
    """ enumerates all binary trees formed form segmentations """

    def __init__(self):
        self.string_trees = set([])
        self.trees = set([])
        
    def enumerate_segmentations(self, string):
        """ enumerate segmentations """
        N = len(string)
        segs = []
        for bits in it.product([0, 1], repeat=N-1):
            i, j = 0, 0
            seg = []
            for k, b in enumerate((0,)+bits):
                j = k
                if b == 1:
                    seg.append((string[i:j], i, j))
                    i = j
            seg.append((string[i:j+1], i, j+1))
            segs.append(seg)
        return segs

    def enumerate_labelings(self, seg, K):
        """ labelings """
        labeled_seg = []
        N = len(seg)
        for label in it.product(range(K), repeat=N):
            new = []
            for s, l in zip(seg, label):
                new.append(s + (l,))
            if len(new) == 1 and new[0][-1] != 0:
                continue
            labeled_seg.append(new)
        return labeled_seg
        
    def merge(self, nodes, i):
        """ merges adjacent nodes and forms a new list """
        assert i < len(nodes)-1
        new_nodes = []
        for j, node in enumerate(nodes):
            if i == j:
                node1, node2 = nodes[j], nodes[j+1]
                new_nodes.append(NonTerminal(node1, node2, node1.i, node2.j, 0))
            elif i+1 == j:
                continue
            else:
                new_nodes.append(node)
        return new_nodes

    def go(self, lst):
        return self.enumerate_trees([Terminal(*x) for x in lst])
    
    def enumerate_trees(self, nodes):
        """ enumerate_trees all trees """
        if len(nodes) == 1:
            tree = self.display(nodes[0])
            if tree not in self.string_trees:
                self.string_trees.add(tree)
                self.trees.add(nodes[0])
        else:
            for i in xrange(len(nodes)-1):
                new_nodes = self.merge(nodes, i)
                self.enumerate_trees(new_nodes)

    def display(self, node):
        """ print the tree """
        if isinstance(node, Terminal):
            return str(node)
        else:
            string = "( "
            string += self.display(node.node1)
            string += " "
            string += self.display(node.node2)
            string += " )"
            return string

    def walk(self, node, string):
        """ walk the tree """
        yield (node.i, node.j, node.label)
        if isinstance(node, NonTerminal):
            for x in self.walk(node.node1, string):
                yield x
            for x in self.walk(node.node2, string):
                yield x
        
if __name__ == "__main__":
    G = 2
    master = "abcdefghijklmnopqrstuvwxyz"
    for size in xrange(3, 4):
        tree_enumerator = Enumerator()
        string = master[:size]
        for seg in sorted(tree_enumerator.enumerate_segmentations(string), key=lambda x: len(x)):
            for lab in tree_enumerator.enumerate_labelings(seg, G):
                tree_enumerator.go(lab)

        print len(tree_enumerator.trees)
        
    # HOW TO DISPLAY TREES + compute probabilities 
    for root in tree_enumerator.trees:
    #    print tree_enumerator.display(node)
        for node in tree_enumerator.walk(root, string):
            print node
        print
