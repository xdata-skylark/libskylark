#
# Find LIBHDFS includes and library
#
# HDFS
#
# It can be found at:
#
# LIBHDFS_INCLUDE_DIR - where to find hdfs.h
# LIBHDFS_LIBRARY     - qualified libraries to link against.
# LIBHDFS_FOUND       - do not attempt to use if "no" or undefined.

FIND_PATH(LIBHDFS_INCLUDE_DIR hdfs.h
  $ENV{LIBHDFS_ROOT}/include
  $ENV{HOME}/.local/include
  $ENV{HOME}/local/include
  $ENV{HOME}/Software/include
  /usr/local/include
  /usr/include
  NO_DEFAULT_PATH
)

FIND_LIBRARY(LIBHDFS_LIBRARY hdfs
  $ENV{LIBHDFS_ROOT}/lib/native
  $ENV{HOME}/.local/lib/
  $ENV{HOME}/local/lib/
  $ENV{HOME}/Software/lib/
  /usr/local/lib/
  /usr/lib/
  NO_DEFAULT_PATH
)



IF(LIBHDFS_INCLUDE_DIR AND LIBHDFS_LIBRARY)
  set(LIBHDFS_FOUND "YES")
  set(LIBHDFS_LIBRARIES
    ${LIBHDFS_LIBRARY}
  )
ENDIF(LIBHDFS_INCLUDE_DIR AND LIBHDFS_LIBRARY)

IF (LIBHDFS_FOUND)
  IF (NOT LIBHDFS_FIND_QUIETLY)
    MESSAGE(STATUS "Found LIBHDFS: ${LIBHDFS_LIBRARIES}")
  ENDIF (NOT LIBHDFS_FIND_QUIETLY)
ELSE (LIBHDFS_FOUND)
  IF (LIBHDFS_FIND_REQUIRED)
    MESSAGE(STATUS "LIBHDFS not found!")
  ENDIF (LIBHDFS_FIND_REQUIRED)
ENDIF (LIBHDFS_FOUND)
