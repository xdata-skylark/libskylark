#
# Find Elemental includes and library
#
# Elemental
# It can be found at:
#
# Elemental_INCLUDE_DIR - where to find H5hut.h
# Elemental_LIBRARY     - qualified libraries to link against.
# Elemental_FOUND       - do not attempt to use if "no" or undefined.

FIND_PATH(Elemental_INCLUDE_DIR El/core.hpp
  /usr/local/include
  /usr/include
  $ENV{HOME}/Software/include
  $ENV{ELEMENTAL_ROOT}/include
)

FIND_LIBRARY(Elemental_LIBRARY El
   /usr/local/lib
   /usr/lib
   $ENV{HOME}/Software/lib
   $ENV{ELEMENTAL_ROOT}/lib
)

FIND_LIBRARY(Pmrrr_LIBRARY pmrrr
   /usr/local/lib
   /usr/lib
   $ENV{HOME}/Software/lib
   $ENV{ELEMENTAL_ROOT}/lib
)

IF(Elemental_INCLUDE_DIR AND Elemental_LIBRARY AND Pmrrr_LIBRARY)
  SET( Elemental_FOUND "YES")
ENDIF(Elemental_INCLUDE_DIR AND Elemental_LIBRARY AND Pmrrr_LIBRARY)

IF (Elemental_FOUND)
  IF (NOT Elemental_FIND_QUIETLY)
    MESSAGE(STATUS
            "Found Elemental:${Elemental_LIBRARY}")
    MESSAGE(STATUS
            "Found pmrrr:${Pmrrr_LIBRARY}")
  ENDIF (NOT Elemental_FIND_QUIETLY)
ELSE (Elemental_FOUND)
  IF (Elemental_FIND_REQUIRED)
    MESSAGE(STATUS "Elemental not found!")
  ENDIF (Elemental_FIND_REQUIRED)
ENDIF (Elemental_FOUND)
