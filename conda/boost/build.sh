#!/bin/bash

# sources:
# https://github.com/ContinuumIO/anaconda-recipes/blob/master/boost/build.sh
# https://github.com/conda-forge/boost-feedstock/blob/master/recipe/build.sh
# https://github.com/menpo/conda-boost/blob/master/conda/build.sh

INCLUDE_PATH="${PREFIX}/include"
LIBRARY_PATH="${PREFIX}/lib"
CXXFLAGS="${CXXFLAGS} -fPIC -std=c++11"
LINKFLAGS="${LINKFLAGS} -std=c++11 -L${LIBRARY_PATH}"

./bootstrap.sh \
--prefix="${PREFIX}" \
--with-libraries=mpi,serialization,program_options,filesystem,system

./b2 \
runtime-link=shared \
link=static,shared \
include="${INCLUDE_PATH}" \
linkflags="${LINKFLAGS}" \
-j"${CPU_COUNT}" \
--user-config="${RECIPE_DIR}/user-config.jam" \
install
