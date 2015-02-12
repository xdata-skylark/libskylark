#
# Find Random123 includes
#
# Random123
# It can be found at:
#
# Random123_INCLUDE_DIR - where to find Random123.h
# Random123_FOUND       - boolean indicating if Random123 was found.

FIND_PATH(Random123_INCLUDE_DIR Random123/threefry.h Random123/MicroURNG.hpp
  $ENV{RANDOM123_ROOT}/include
  $ENV{HOME}/Software/include
  /usr/local/include
  /usr/include
  NO_DEFAULT_PATH
)

IF(Random123_INCLUDE_DIR)
  SET( Random123_FOUND "YES")
ENDIF(Random123_INCLUDE_DIR)

IF (Random123_FOUND)
  IF (NOT Random123_FIND_QUIETLY)
    MESSAGE(STATUS "Found Random123:${Random123_INCLUDE_DIR}")
  ENDIF (NOT Random123_FIND_QUIETLY)
ELSE (Random123_FOUND)
  IF (Random123_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Random123 not found!")
  ENDIF (Random123_FIND_REQUIRED)
ENDIF (Random123_FOUND)
