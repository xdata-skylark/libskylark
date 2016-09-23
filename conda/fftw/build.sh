#!/bin/bash

NPROC=`nproc`

export CC="${PREFIX}/bin/gcc"

./configure \
--prefix="${PREFIX}" \
--enable-shared

make -j "${NPROC}"

make install
