#!/bin/bash

NPROC=`nproc`

export CC="${PREFIX}/bin/gcc"
export CFLAGS="-fPIC -fopenmp"

./configure \
--prefix="${PREFIX}" \
--enable-RAM=16000 \
--enable-DDL \
--enable-IL \
--enable-PARA=8

make -j "${NPROC}"

make install

