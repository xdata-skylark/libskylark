# Find Elemental
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")
find_package(Elemental REQUIRED)
if (Elemental_FOUND)
  include_directories (${Elemental_INCLUDE_DIR})
  link_directories (${Elemental_LIBRARY_DIR})

  # check if elemental build type is Hybrid
  file(STRINGS "${Elemental_INCLUDE_DIR}/El/config.h"
    ELEMENTAL_HYBRID_BUILD_TYPE REGEX "^[ \t]*#define[ \t]+EL_HYBRID")

  if(ELEMENTAL_HYBRID_BUILD_TYPE)
    message(STATUS "Elemental was built in hybrid mode. Enabling hybrid support as well.")
    set (USE_HYBRID
         1
         CACHE
         STRING
         "Enables use of hybrid MPI/OpenMP parallelization"
         FORCE)
  endif(ELEMENTAL_HYBRID_BUILD_TYPE)
endif (Elemental_FOUND)

# 3.1 Elemental requires BLAS and LAPACK libraries
if(NOT MATH_LIBS)
  set(BLAS_LIBRARIES $ENV{BLAS_LIBRARIES})
  set(LAPACK_LIBRARIES $ENV{LAPACK_LIBRARIES})
  if(NOT BLAS_LIBRARIES OR NOT LAPACK_LIBRARIES)
    find_package(BLAS)
    find_package(LAPACK)
  endif(NOT BLAS_LIBRARIES OR NOT LAPACK_LIBRARIES)

  if(NOT BLAS_LIBRARIES OR NOT LAPACK_LIBRARIES)
      message(FATAL_ERROR "Elemental needs BLAS and LAPACK")
  endif(NOT BLAS_LIBRARIES OR NOT LAPACK_LIBRARIES)
  set (Elemental_LIBRARY
    ${Elemental_LIBRARY}
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES}
  )
else()
  set (Elemental_LIBRARY
    ${Elemental_LIBRARY}
    ${MATH_LIBS}
  )
endif()
