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
  /usr/local/include
  /usr/include
  $ENV{HOME}/Software/include
  $ENV{FFTW_ROOT}/include
)

FIND_LIBRARY(FFTW_LIBRARY fftw3
   /usr/local/lib
   /usr/lib
   $ENV{HOME}/Software/lib
   $ENV{FFTW_ROOT}/lib
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
  ENDIF (FFTW_FIND_REQUIRED)
ENDIF (FFTW_FOUND)
