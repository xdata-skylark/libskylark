from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

if not os.path.exists('randgen.pyx'):
    print 'Generating randgen.pyx'
    execfile('generate_randgen.py')

if not os.path.exists('test_randgen.py'):
    print 'Generating test_randgen.py'
    execfile('generate_test_randgen.py')

extensions = [
    Extension('randgen', 
              ['randgen.pyx'],
              include_dirs = [os.path.join(os.environ['RANDOM123_ROOT'], 'include'),
                              os.path.join(os.environ['INSTALL_DIR'], 'include')],
              extra_compile_args = ['-D__STDC_CONSTANT_MACROS'],
              language = 'c++'),
]


setup(
  name = 'randgen',
  ext_modules = cythonize(extensions)
)
