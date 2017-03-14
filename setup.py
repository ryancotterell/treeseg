import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize


setup(name='segmenter',
      version='1.0',
      description='',
      author='Ryan Cotterell',
      packages=['segmenter'],
      install_requires=[
          'pyenchant>=1.6.6',
          'termcolor>=1.1.0',
          'python-Levenshtein>=0.12.0'
      ],
      include_dirs=[np.get_include(),
                    #'/usr/local/include/boost',
                    os.path.expanduser('~/anaconda/include/')],
      library_dirs = ['/usr/local/lib'],
      ext_modules = cythonize(['segmenter/**/*.pyx']))
