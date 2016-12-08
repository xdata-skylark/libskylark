cmake_minimum_required (VERSION 2.8.3)

# we don't allow libSkylark to be build in the source directory
set (CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project (SKYLARK)

set (SKYLARK_VERSION_MAJOR 0)
set (SKYLARK_VERSION_MINOR 14)

# Unfortunately we need Fortran ;(
enable_language( Fortran )


#-----------------------------------------------------------------------------
# config.h
include (${CMAKE_SOURCE_DIR}/CMake/cmake_git_version.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_date.cmake)

set (PACKAGE \"LIBSKYLARK\")
set (PACKAGE_BUGREPORT \"klclarks@us.ibm.com\")
set (PACKAGE_NAME \"LIBSKYLARK\")
set (PACKAGE_STRING \"LIBSKYLARK\")
set (PACKAGE_VERSION \"0.14\")
set (VERSION \"0.14\")


#--------------------------------------------------------------------------
# Flags
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
       "Choose the type of build, options are: None Debug Release
        RelWithDebInfo HybridRel HybridRelWithDebInfo."
        FORCE
      )
endif (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

message (STATUS "Build type is: " ${CMAKE_BUILD_TYPE})

# Select compiler dependent flags.
set (CMAKE_CXX_FLAGS_HYBRIDRELWITHDEBINFO "-O3 -g -fopenmp")
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set (CMAKE_CXX_FLAGS_HYBRIDREL "-O3 -fopenmp")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")
set (CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")

include (${CMAKE_SOURCE_DIR}/CMake/cmake_compiler_flags.cmake)

message (STATUS "Building on ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR} on
                 $ENV{HOSTNAME}")
set (CMAKE_CXX_FLAGS "${COMPILER_SPEC_FLAGS} ${CMAKE_CXX_FLAGS}")


#--------------------------------------------------------------------------
# Includes
# Add the include directory path for SKYLARK. Remember that only the top dir
# is added for the include, meaning that all the subdirs have to be explictly
# referred. For example,
# #include "../utility/dense_1D.hpp"
# is used in sketch/ subdirectory.
include_directories (${CMAKE_SOURCE_DIR})
include_directories (${CMAKE_BINARY_DIR})


#--------------------------------------------------------------------------
# MPI compiler
find_package (MPI REQUIRED)
#set (CMAKE_CXX_COMPILER ${MPI_COMPILER})
include_directories (${MPI_CXX_INCLUDE_PATH})
set (SKYLARK_LIBS
    ${SKYLARK_LIBS}
    ${MPI_CXX_LIBRARIES})

if (MPI_CXX_COMPILE_FLAGS)
  set (CMAKE_CXX_FLAGS "${MPI_CXX_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
endif (MPI_CXX_COMPILE_FLAGS)

if (MPI_CXX_LINK_FLAGS)
  #FIXME: add "${MPI_CXX_LINK_FLAGS}"
endif (MPI_CXX_LINK_FLAGS)


#---------------------------------------------------------------------------
# dependent packages
include (${CMAKE_SOURCE_DIR}/CMake/cmake_boost.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_elemental.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_random123.cmake)

# optional packages
include (${CMAKE_SOURCE_DIR}/CMake/cmake_hdf5.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_hdfs.cmake)


#-----------------------------------------------------------------------------
# options

#FIXME that should be optional packages and not options
include (${CMAKE_SOURCE_DIR}/CMake/cmake_fftw_option.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_kissfft_option.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_spiralwht_option.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_combblas_option.cmake)

include (${CMAKE_SOURCE_DIR}/CMake/cmake_hybrid_option.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_profiler_option.cmake)

# collect all optional libs
set (OPTIONAL_LIBS "")
include (${CMAKE_SOURCE_DIR}/CMake/cmake_optional_libs.cmake)


#-----------------------------------------------------------------------------
# Toggle different build options

include (${CMAKE_SOURCE_DIR}/CMake/cmake_build_capi.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_build_python_interface.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_build_nla.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_build_ml.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_build_examples.cmake)

option (BUILD_TESTS "Whether we should build the tests" OFF)
if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif (BUILD_TESTS)

include (${CMAKE_SOURCE_DIR}/CMake/cmake_documentation.cmake)

include (${CMAKE_SOURCE_DIR}/CMake/cmake_find_module.cmake)
include (${CMAKE_SOURCE_DIR}/CMake/cmake_skylark_install.cmake)

