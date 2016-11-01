#
# Find KISSFFT includes and library
#
# KISSFFT
# It can be found at:
#
# KISSFFT_INCLUDE_DIR - where to find KISSFFT.
# KISSFFT_LIBRARY     - qualified libraries to link against. Maybe it's not necesary
# KISSFFT_FOUND       - do not attempt to use if "no" or undefined.

FIND_PATH(KISSFFT_INCLUDE_DIR kissfft.hh
  $ENV{KISSFFT_ROOT}/
  $ENV{HOME}/.local/include
  $ENV{HOME}/local/include
  $ENV{HOME}/Software/include
  /usr/local/include
  /usr/include
  NO_DEFAULT_PATH
)


IF(KISSFFT_INCLUDE_DIR)
  SET( KISSFFT_FOUND "YES")
ENDIF(KISSFFT_INCLUDE_DIR)


IF (KISSFFT_FOUND)
  MESSAGE(STATUS
          "Found KissFFT:${KISSFFT_INCLUDE_DIR}")
ELSE (KISSFFT_FOUND)
  MESSAGE(STATUS "Warning: KissFFTW not found.")
ENDIF (KISSFFT_FOUND)



# FIND_LIBRARY(FFTWF_LIBRARY fftw3f
#   $ENV{FFTW_ROOT}/lib
#   $ENV{HOME}/.local/lib
#   $ENV{HOME}/local/lib
#   $ENV{HOME}/Software/lib
#   /usr/local/lib
#   /usr/lib
#   NO_DEFAULT_PATH
# )

# IF(KISSFFT_INCLUDE_DIR AND FFTWF_LIBRARY)
#  SET( KISSFFT_FOUND "YES")
# ENDIF(KISSFFT_INCLUDE_DIR AND FFTWF_LIBRARY)

# IF (FFTWF_FOUND)
#  IF (NOT FFTW_FIND_QUIETLY)
#    MESSAGE(STATUS
#            "Found FFTW single precision:${FFTWF_LIBRARY}")
#  ENDIF (NOT FFTW_FIND_QUIETLY)
# ELSE (FFTWF_FOUND)
#  MESSAGE(STATUS "Warning: FFTW single precision not found.")
# ENDIF (FFTWF_FOUND)
