#!/usr/bin/env bash

# To build skylark (with CombBLAS support):
yes | git clone https://github.com/xdata-skylark/libskylark.git

mkdir ${SKYLARK_BUILD_DIR}
mkdir ${SKYLARK_INSTALL_DIR}
cd ${SKYLARK_BUILD_DIR}
CXX=mpicc cmake -DCMAKE_INSTALL_PREFIX=${SKYLARK_INSTALL_DIR} -DUSE_COMBBLAS=ON ${SKYLARK_SRC_DIR}
make
make install
