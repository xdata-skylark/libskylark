#
# Find FFTW single precision includes and library
#
# FFTWF
# It can be found at:
#
# FFTWF_INCLUDE_DIR - where to find FFTW single precision.
# FFTWF_LIBRARY     - qualified libraries to link against.
# FFTWF_FOUND       - do not attempt to use if "no" or undefined.

FIND_PATH(FFTWF_INCLUDE_DIR fftw3.h
  $ENV{FFTW_ROOT}/include
  $ENV{HOME}/.local/include
  $ENV{HOME}/local/include
  $ENV{HOME}/Software/include
  /usr/local/include
  /usr/include
  NO_DEFAULT_PATH
)

FIND_LIBRARY(FFTWF_LIBRARY fftw3f
   $ENV{FFTW_ROOT}/lib
   $ENV{HOME}/.local/lib
   $ENV{HOME}/local/lib
   $ENV{HOME}/Software/lib
   /usr/local/lib
   /usr/lib
   NO_DEFAULT_PATH
)

IF(FFTWF_INCLUDE_DIR AND FFTWF_LIBRARY)
  SET( FFTWF_FOUND "YES")
ENDIF(FFTWF_INCLUDE_DIR AND FFTWF_LIBRARY)

IF (FFTWF_FOUND)
  IF (NOT FFTW_FIND_QUIETLY)
    MESSAGE(STATUS
            "Found FFTW single precision:${FFTWF_LIBRARY}")
  ENDIF (NOT FFTW_FIND_QUIETLY)
ELSE (FFTWF_FOUND)
  MESSAGE(STATUS "Warning: FFTW single precision not found.")
ENDIF (FFTWF_FOUND)
