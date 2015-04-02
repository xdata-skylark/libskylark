#
# Find HDF5 includes and library
#
# HDF5
#
# It can be found at:
#
# HDF5_INCLUDE_DIR - where to find hdf5.h
# HDF5_LIBRARY     - qualified libraries to link against.
# HDF5_FOUND       - do not attempt to use if "no" or undefined.

FIND_PATH(HDF5_INCLUDE_DIR hdf5.h
  $ENV{HDF5_ROOT}/include
  $ENV{HOME}/.local/include
  $ENV{HOME}/local/include
  $ENV{HOME}/Software/include
  /usr/local/include
  /usr/include
  NO_DEFAULT_PATH
)

FIND_LIBRARY(HDF5_LIBRARY hdf5
  $ENV{HDF5_ROOT}/lib
  $ENV{HOME}/.local/lib
  $ENV{HOME}/local/lib
  $ENV{HOME}/Software/lib
  /usr/local/lib
  /usr/lib
  NO_DEFAULT_PATH
)

FIND_LIBRARY(HDF5_CPP_LIBRARY hdf5
  $ENV{HDF5_ROOT}/lib
  $ENV{HOME}/.local/lib
  $ENV{HOME}/local/lib
  $ENV{HOME}/Software/lib
  /usr/local/lib
  /usr/lib
  NO_DEFAULT_PATH
)

IF(HDF5_INCLUDE_DIR AND HDF5_LIBRARY AND HDF5_CPP_LIBRARY)
  set(HDF5_FOUND "YES")
  set(HDF5_LIBRARIES
    ${HDF5_LIBRARY}
    ${HDF5_CPP_LIBRARY}
  )
ENDIF(HDF5_INCLUDE_DIR AND HDF5_LIBRARY AND HDF5_CPP_LIBRARY)

IF (HDF5_FOUND)
  IF (NOT HDF5_FIND_QUIETLY)
    MESSAGE(STATUS "Found HDF5: ${HDF5_LIBRARIES}")
  ENDIF (NOT HDF5_FIND_QUIETLY)
ELSE (HDF5_FOUND)
  IF (HDF5_FIND_REQUIRED)
    MESSAGE(STATUS "HDF5 not found!")
  ENDIF (HDF5_FIND_REQUIRED)
ENDIF (HDF5_FOUND)
