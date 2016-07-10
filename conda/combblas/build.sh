#!/bin/bash

NPROC=`nproc`

echo """
diff --git a/RefGen21.h b/RefGen21.h
index b8c7974..f93592c 100644
--- a/RefGen21.h
+++ b/RefGen21.h
@@ -134,7 +134,7 @@ public:

        /* 32-bit code */
        uint32_t h = (uint32_t)(x >> 32);
-       uint32_t l = (uint32_t)(x & UINT32_MAX);
+       uint32_t l = (uint32_t)(x & std::numeric_limits<uint32_t>::max());
        #ifdef USE_GCC_BYTESWAP
         h = __builtin_bswap32(h);
         l = __builtin_bswap32(l);
""" | git apply --ignore-space-change --ignore-whitespace

CC="${PREFIX}/bin/mpicc -cc=${PREFIX}/bin/gcc" CXX="${PREFIX}/bin/mpicxx -cxx=${PREFIX}/bin/g++" \
cmake -DBUILD_SHARED_LIBS=ON .

make -j "${NPROC}"

cp *.so "${PREFIX}/lib"
mkdir "${PREFIX}/include/CombBLAS"
cp *.h "${PREFIX}/include/CombBLAS"
cp *.cpp "${PREFIX}/include/CombBLAS"
cp -R SequenceHeaps "${PREFIX}/include/CombBLAS"
cp -R psort-1.0 "${PREFIX}/include/CombBLAS"
cp -R graph500-1.2 "${PREFIX}/include/CombBLAS"


