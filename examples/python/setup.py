"--------------------------------------------------------------------------"
" Set up the libraries to import "
"--------------------------------------------------------------------------"

" 1. Import ctypes and manually load OpenMPI "
import sys
import ctypes

if sys.platform == 'darwin':
  mpi = ctypes.CDLL('libmpi.1.dylib', ctypes.RTLD_GLOBAL)
else:
  #FIXME: do we need RTLD_GLOBAL for Linux?
  try:
    mpi = ctypes.CDLL('libmpi.so.0', ctypes.RTLD_GLOBAL)
  except OSError:
    mpi = ctypes.CDLL('libmpi.so', ctypes.RTLD_GLOBAL)

" 2. Load the required libraries "
import numpy
import mpi
from skylark import utility
from skylark import sketch 
"--------------------------------------------------------------------------"
