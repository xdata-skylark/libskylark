#
# Find CombBLAS includes and library
#
# CombBLAS
# It can be found at:
#
# CombBLAS_INCLUDE_DIR - where to find CombBLAS.h
# CombBLAS_LIBRARY     - qualified libraries to link against.
# CombBLAS_FOUND       - do not attempt to use if "no" or undefined.

FIND_PATH(CombBLAS_INCLUDE_DIR CombBLAS/CombBLAS.h CombBLAS/SpParMat.h CombBLAS/SpParVec.h CombBLAS/DenseParVec.h
  /usr/local/include
  /usr/include
  $ENV{HOME}/Software/include
  $ENV{COMBBLAS_ROOT}/include
)

find_library(MPITypelib_LIBRARY MPITypelib
  /usr/local/lib
  /usr/lib
  $ENV{HOME}/Software/lib
  $ENV{COMBBLAS_ROOT}/lib
)
find_library(CommGridlib_LIBRARY CommGridlib
  /usr/local/lib
  /usr/lib
  $ENV{HOME}/Software/lib
  $ENV{COMBBLAS_ROOT}/lib
)
find_library(MemoryPoollib_LIBRARY MemoryPoollib
  /usr/local/lib
  /usr/lib
  $ENV{HOME}/Software/lib
  $ENV{COMBBLAS_ROOT}/lib
)

IF(MPITypelib_LIBRARY AND CommGridlib_LIBRARY AND MemoryPoollib_LIBRARY)
  set(CombBLAS_LIBRARIES
    ${MPITypelib_LIBRARY}
    ${CommGridlib_LIBRARY}
    ${MemoryPoollib_LIBRARY}
  )
  set(CombBLAS_LIBRARY "YES")
ENDIF(MPITypelib_LIBRARY AND CommGridlib_LIBRARY AND MemoryPoollib_LIBRARY)


IF(CombBLAS_INCLUDE_DIR AND CombBLAS_LIBRARY)
  SET( CombBLAS_FOUND "YES")
ENDIF(CombBLAS_INCLUDE_DIR AND CombBLAS_LIBRARY)

IF (CombBLAS_FOUND)
  IF (NOT CombBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found CombBLAS:${CombBLAS_LIBRARIES}")
  ENDIF (NOT CombBLAS_FIND_QUIETLY)
ELSE (CombBLAS_FOUND)
  IF (CombBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "CombBLAS not found!")
  ENDIF (CombBLAS_FIND_REQUIRED)
ENDIF (CombBLAS_FOUND)
