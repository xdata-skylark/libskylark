#!/bin/sh

NPROC=`nproc`

git checkout b9f9d6702bb3f176d0fc2c29268f24b1c01ac9e8

export BLAS_LIBRARIES="-L${PREFIX} -lopenblas -lm"
export LAPACK_LIBRARIES="-L${PREFIX} -lopenblas -lm"
export BOOST_ROOT="${PREFIX}"
export RANDOM123_ROOT="${PREFIX}"
export ELEMENTAL_ROOT="${PREFIX}"
export COMBBLAS_ROOT="${PREFIX}"


CC="${PREFIX}/bin/mpicc -cc=${PREFIX}/bin/gcc" CXX="${PREFIX}/bin/mpicxx -cxx=${PREFIX}/bin/g++" \
cmake \
-DCMAKE_INSTALL_PREFIX="${PREFIX}" \
-DUSE_FFTW=OFF \
-DUSE_COMBBLAS=ON \
.

make -j "${NPROC}"

make install

