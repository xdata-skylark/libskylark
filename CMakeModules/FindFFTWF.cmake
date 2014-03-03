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
  /usr/local/include
  /usr/include
  $ENV{HOME}/Software/include
  $ENV{FFTW_ROOT}/include
)

FIND_LIBRARY(FFTWF_LIBRARY fftw3f
   /usr/local/lib
   /usr/lib
   $ENV{HOME}/Software/lib
   $ENV{FFTW_ROOT}/lib
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
  IF (FFTW_FIND_REQUIRED)
    MESSAGE(STATUS "FFTW single precision not found!")
  ENDIF (FFTW_FIND_REQUIRED)
ENDIF (FFTWF_FOUND)
