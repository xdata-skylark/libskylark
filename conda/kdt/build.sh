#!/bin/bash

export CC="${PREFIX}/bin/mpicc -cc=${PREFIX}/bin/gcc" 
export CXX="${PREFIX}/bin/mpicxx -cxx=${PREFIX}/bin/g++"

$PYTHON setup.py build
$PYTHON setup.py install

