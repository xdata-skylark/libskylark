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
  $ENV{KISSFFT_ROOT}
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
