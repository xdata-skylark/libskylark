#
# Find FFTW includes and library
#
# FFTW
# It can be found at:
#
# FFTW_INCLUDE_DIR - where to find FFTW.
# FFTW_LIBRARY     - qualified libraries to link against.
# FFTW_FOUND       - do not attempt to use if "no" or undefined.

FIND_PATH(FFTW_INCLUDE_DIR fftw3.h
  $ENV{FFTW_ROOT}/include
  $ENV{HOME}/.local/lib
  $ENV{HOME}/local/lib
  $ENV{HOME}/Software/include
  /usr/local/include
  /usr/include
  NO_DEFAULT_PATH
)

FIND_LIBRARY(FFTW_LIBRARY fftw3
   $ENV{FFTW_ROOT}/lib
   $ENV{HOME}/.local/lib
   $ENV{HOME}/local/lib
   $ENV{HOME}/Software/lib
   /usr/local/lib
   /usr/lib
   /usr/lib/x86_64-linux-gnu
   NO_DEFAULT_PATH
)

IF(FFTW_INCLUDE_DIR AND FFTW_LIBRARY)
  SET( FFTW_FOUND "YES")
ENDIF(FFTW_INCLUDE_DIR AND FFTW_LIBRARY)

IF (FFTW_FOUND)
  IF (NOT FFTW_FIND_QUIETLY)
    MESSAGE(STATUS
            "Found FFTW:${FFTW_LIBRARY}")
  ENDIF (NOT FFTW_FIND_QUIETLY)
ELSE (FFTW_FOUND)
  IF (FFTW_FIND_REQUIRED)
    MESSAGE(STATUS "FFTW not found!")
  ELSE(FFTW_FIND_REQUIRED)
    MESSAGE(STATUS "Warning: FFTW not found.")
  ENDIF (FFTW_FIND_REQUIRED)
ENDIF (FFTW_FOUND)
