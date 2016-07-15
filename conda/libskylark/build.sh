#!/bin/sh

NPROC=`nproc`

# skylark branch
git checkout development

export BLAS_LIBRARIES="-L${PREFIX} -lopenblas -lm"
export LAPACK_LIBRARIES="-L${PREFIX} -lopenblas -lm"
export BOOST_ROOT="${PREFIX}"
export RANDOM123_ROOT="${PREFIX}"
export ELEMENTAL_ROOT="${PREFIX}"
export FFTW_ROOT="${PREFIX}"


CC="${PREFIX}/bin/mpicc -cc=${PREFIX}/bin/gcc" CXX="${PREFIX}/bin/mpicxx -cxx=${PREFIX}/bin/g++" \
cmake \
-DCMAKE_INSTALL_PREFIX="${PREFIX}" \
-DUSE_FFTW=ON \
-DUSE_COMBBLAS=OFF \
.

make -j "${NPROC}"

make install

