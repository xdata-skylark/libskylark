#
# Find CombBLAS includes and library
#
# CombBLAS
# It can be found at:
#
# CombBLAS_INCLUDE_DIR - where to find CombBLAS.h
# CombBLAS_LIBRARY     - qualified libraries to link against.
# CombBLAS_FOUND       - do not attempt to use if "no" or undefined.

FIND_PATH(CombBLAS_INCLUDE_DIR CombBLAS.h SpParMat.h SpParVec.h DenseParVec.h
  $ENV{COMBBLAS_ROOT}
  $ENV{HOME}/.local/include
  $ENV{HOME}/local/include
  $ENV{HOME}/Software/include
  /usr/local/include
  /usr/local/include/CombBLAS
  /usr/include
  NO_DEFAULT_PATH
)

find_library(MPITypelib_LIBRARY MPITypelib
  $ENV{COMBBLAS_ROOT}
  $ENV{HOME}/.local/lib
  $ENV{HOME}/local/lib
  $ENV{HOME}/Software/lib
  /usr/local/lib
  /usr/lib
  NO_DEFAULT_PATH
)

find_library(CommGridlib_LIBRARY CommGridlib
  $ENV{COMBBLAS_ROOT}
  $ENV{HOME}/.local/lib
  $ENV{HOME}/local/lib
  $ENV{HOME}/Software/lib
  /usr/local/lib
  /usr/lib
  NO_DEFAULT_PATH
)

find_library(MemoryPoollib_LIBRARY MemoryPoollib
  $ENV{COMBBLAS_ROOT}
  $ENV{HOME}/.local/lib
  $ENV{HOME}/local/lib
  $ENV{HOME}/Software/lib
  /usr/local/lib
  /usr/lib
  NO_DEFAULT_PATH
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
