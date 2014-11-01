#!/usr/bin/env python

from distutils.core import setup

setup(name='skylark',
      version='0.1',
      description='libSkylark: Sketching-based Matrix Computations for Machine Learninge',
      author='IBM Corporation, Reseach Division',
      author_email='vsindhw@us.ibm.com',
      url='http://xdata-skylark.github.io/libskylark/',
      packages=['skylark'],
      package_dir = {'': 'python'}
     )
