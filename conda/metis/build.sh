#!/bin/bash

NPROC=`nproc`

export CC="${PREFIX}/bin/gcc"

make config shared=1 prefix="${PREFIX}"

make -j "${NPROC}"

make install

