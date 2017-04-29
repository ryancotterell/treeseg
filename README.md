python segmenter/manager.py --train data/splits/train1 --dev data/splits/dev1 --test data/splits/test1 --type chunk --decode all --log test.log --T_C 0.00000001

Codebase used for the following publications:

1) Ryan Cotterell, Arun Kumar and Hinrich Schütze. Morphological Segmentation Inside-Out. EMNLP 2016.

2) Ryan Cotterell and Hinrich Schütze. Joint Semantic Synthesis and Morphological Analysis of the Derived Word. TACL. 2017. arXiv

3) Ryan Cotterell, Tim Vieira and Hinrich Schütze. A Joint Model of Orthography and Morphological Segmentation. NAACL. 2016.

As regards 3), the code presented here is a substantially revised
codebase of the ones used to obtain the experimental results. Due
to better optimization, the results have slightly improved! 