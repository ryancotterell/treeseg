import codecs
import sys

PREFIX = "prefix"
SUFFIX = "suffix"

def display(node):
    """ displays the tree """
    if len(node.cdr) == 0:
        return str(node)
    return "("+str(node) + " " + " ".join(map(display, node.cdr))+")"

def walk(node):
    if len(node.cdr) != 0:
        for child in node.cdr:
            yield child
            for x in walk(child):
                yield x

def string(node):
    """ get the yield of the tree """
    if len(node.cdr) == 0:
        word, label = str(node.car).split(":")
        return word
    rest = ""
    for child in node.cdr:
        rest += string(child)
    return rest

def segmentation(node):
    """ gets the segmentation from the tree """
    if len(node.cdr) == 0:
        word, label = str(node.car).split(":")
        l = 2
        if label == PREFIX:
            l = 0
        elif label == SUFFIX:
            l = 1
        return [(word, l)]
    rest = []
    for child in node.cdr:
        rest += segmentation(child)
    return rest
                
class Node(object):
    """ tree node """
    def __init__(self, car):
        self.car = car
        self.cdr = []
        self.i, self.j = 0, 0
        self.word, self.label = None, 0

    def __str__(self):
        return unicode(self) + "(" + str(self.i) + "," + str(self.j) + ")"
    
    def __unicode__(self):
        return unicode(self.car)

    def __repr__(self):
        return unicode(self)

class Data(object):
    """ data sources """

    def __init__(self, fname):
        self.fname = fname
        self.read()
        for _, (tree, _) in self:
            self.annotate(tree)

    def read(self):
        """ read in the trees """
        self.srs = []
        self.trees = []
        self.segments = []
        # load trees
        with codecs.open(self.fname, 'rb', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                split = line.split(" ")
                sr = split[0]
                self.srs.append(sr)
                line = " ".join(split[1:])
                root = Node('ROOT')
                stack = [root]
                self.trees.append(root)
                buffer = ''
                head = False
                for char in line:
                    if char== '(':
                        if head:
                            node = Node(buffer)
                            stack[-1].cdr.append(node)
                            stack.append(node)
                            buffer = ''
                        head = True
                    elif char == ')':
                        if head:
                            node = Node(buffer)
                            stack[-1].cdr.append(node)
                            head = False
                            buffer = ''
                        else:
                            stack.pop()
                    elif char == ' ':
                        pass
                    else:
                        buffer += char
        # extract segmentation
        for tree in self.trees:
            i = 0
            string_yield = string(tree)
            indices, labels = [], []
            for (word, l) in segmentation(tree):
                j = i+len(word)
                indices.append((i, j))
                labels.append(l)
                assert string_yield[i:j] == word
                i = j
            self.segments.append((indices, labels))
        
    def annotate(self, node, i=0):
        """ annotate nodes with the spans """
        if len(node.cdr) == 0:
            word, label = str(node.car).split(":")
            node.i = i
            node.j = i+len(word)
            node.word = word
            if label == PREFIX:
                node.label = 1
            elif label == SUFFIX:
                node.label = 2
            else:
                node.label = 0
        elif len(node.cdr) == 1:
            self.annotate(node.cdr[0], i)
            node.i = node.cdr[0].i
            node.j = node.cdr[0].j
        else:
            self.annotate(node.cdr[0], i)
            node.i = node.cdr[0].i 
            self.annotate(node.cdr[1], node.cdr[0].j)
            node.j = node.cdr[1].j

    def test_annotate(self, node, string):
        if len(node.cdr) == 0:
            word, label = str(node.car).split(":")
            assert word == string[node.i:node.j]
        elif len(node.cdr) == 1:
            self.test_annotate(node.cdr[0], string)
        else:
            self.test_annotate(node.cdr[0], string)
            self.test_annotate(node.cdr[1], string)
            
    def __iter__(self):
        return iter(zip(self.srs, zip(self.trees, self.segments)))
                        
if __name__ == "__main__":
    data = Data(sys.argv[1])
    root = data.trees[0]

    for root in data.trees:
        data.annotate(root)
        data.test_annotate(root, data.string(root))

    print "\n".join([display(x) for x in data.trees[:1]])

