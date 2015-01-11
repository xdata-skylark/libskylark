cmake_minimum_required (VERSION 2.8.3)

# we don't allow Skylark to be build in the source directory
set (CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project (SKYLARK)

set (SKYLARK_VERSION_MAJOR 0)
set (SKYLARK_VERSION_MINOR 1)

# Unfortunately we need Fortran ;(
enable_language( Fortran )

#-----------------------------------------------------------------------------
# config.h

# we start by gathering some revision version information, first we try GIT
execute_process (
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND git rev-parse HEAD
    RESULT_VARIABLE GIT_REPO
    OUTPUT_VARIABLE GIT_REV OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (GIT_REPO EQUAL 0)
  message (STATUS "Building git version ${GIT_REV}")
  set (GIT_REVISION \"${GIT_REV}\")
else (GIT_REPO EQUAL 0)
  message (STATUS "No git repository found.")
  set (GIT_REVISION 0)
endif (GIT_REPO EQUAL 0)

set (BUILD_DATE 0)
EXECUTE_PROCESS(
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND date
    OUTPUT_VARIABLE DATE_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
)
set (BUILD_DATE \"${DATE_OUT}\")


set (PACKAGE \"SKYLARK\")
set (PACKAGE_BUGREPORT \"bla@bla.com\")
set (PACKAGE_NAME \"SKYLARK\")
set (PACKAGE_STRING \"SKYLARK\")
set (PACKAGE_VERSION \"0.1\")
set (VERSION \"0.1\")

#configure_file(config.h.in ${CMAKE_SOURCE_DIR}/config.h)

#--------------------------------------------------------------------------
# Flags
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
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

message(STATUS "Building with ${CMAKE_CXX_COMPILER_ID} compiler")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
        include_directories (
            /usr/llvm-gcc-4.2/lib/gcc/i686-apple-darwin11/4.2.1/include
        )
    endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )

    set (SKYLARK_LIBS
        m
    )

    set (COMPILER_SPEC_FLAGS
        "-W -Wall -Wno-write-strings -Wno-strict-aliasing -Wno-format -Wno-deprecated -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare"
    )

    #set (LINK_FLAGS
    #)

    set (CMAKE_LIB_LINKER_FLAGS  "${CMAKE_LIB_LINKER_FLAGS} -fPIC")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

    set (SKYLARK_LIBS
        m
    )

    set (COMPILER_SPEC_FLAGS
        "-W -Wall -Wno-write-strings -Wno-strict-aliasing -Wno-format -Wno-deprecated -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare"
    )

    #set (LINK_FLAGS
    #)

    set (CMAKE_LIB_LINKER_FLAGS  "${CMAKE_LIB_LINKER_FLAGS} -fPIC")


elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    #TODO: using Intel C++

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")

    set (COMPILER_SPEC_FLAGS
        "-qtune=qp -qarch=qp -qmaxmem=-1 -qcpluscmt -qstrict"
    )
    # -qlanglvl=extended0x

    include_directories (
    )

    link_directories (
        /opt/ibmcmp/xlf/bg/14.1/bglib64/
        /bgsys/drivers/ppcfloor/spi/lib
        /bgsys/drivers/ppcfloor/bgpm/lib/
    )

    set (SKYLARK_LIBS
        xlf90_r
        xlfmath
    )

    set(BGQ_PROFILING_LIBRARIES
        mpihpm
        mpitrace
        bgpm
    )

else()
    message (FATAL "Unsupported compiler!")

endif()

set (CMAKE_CXX_FLAGS "${COMPILER_SPEC_FLAGS} ${CMAKE_CXX_FLAGS}")


message (STATUS "Building on ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR} on
                 $ENV{HOSTNAME}")

# Add the include directory path for SKYLARK. Remember that only the top dir
# is added for the include, meaning that all the subdirs have to be explictly
# referred. For example,
# #include "../utility/dense_1D.hpp"
# is used in sketch/ subdirectory.
include_directories (${CMAKE_SOURCE_DIR})
include_directories (${CMAKE_BINARY_DIR})

#---------------------------------------------------------------------------
# dependent packages
# 1. Find MPI --- we are not adding any include directories here because we
# will be using the mpi compilers, which adds these libraries by default.
find_package (MPI REQUIRED)
set (CMAKE_CXX_COMPILER ${MPI_COMPILER})

# 2. Find Boost with the relevant packages --- Use dynamic boost!
# Without dynamic linking, it's tough to create python bindings.
set (BOOST_ROOT $ENV{BOOST_ROOT})
# turn off system paths if BOOST_ROOT is defined
if (BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS ON)
  set(Boost_NO_BOOST_CMAKE ON)
endif (BOOST_ROOT)

set(BOOST_MIN_VERSION 1.53.0)
find_package (Boost REQUIRED mpi program_options serialization)
if (Boost_FOUND)
  set (SKYLARK_HAVE_BOOST
       1
       CACHE
       STRING
       "Enables use of Boost Libraries"
       FORCE)
  include_directories (${Boost_INCLUDE_DIRS})
  link_directories (${Boost_LIBRARY_DIRS})
  message(STATUS "Found Boost: ${Boost_INCLUDE_DIRS}" )
endif (Boost_FOUND)

# 3. Find Elemental
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")
find_package(Elemental REQUIRED)
if (Elemental_FOUND)
  include_directories (${Elemental_INCLUDE_DIR})
  link_directories (${Elemental_LIBRARY_DIR})

  #FIXME: only enable C++11 if elem >= 0.83?
  #FIXME: properly identify and check flags
  #include(FindCXXFeatures)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")
    set(CXX11_COMPILER_FLAGS "-qlanglvl=extended0x")
  else ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")
    set(CXX11_COMPILER_FLAGS "-std=c++11")
  endif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")
  message(STATUS "CXX11_COMPILER_FLAGS=${CXX11_COMPILER_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CXX11_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")

  # check if elemental build type is Hybrid
  file(STRINGS "${Elemental_INCLUDE_DIR}/elemental/config.h"
    ELEMENTAL_HYBRID_BUILD_TYPE REGEX "^[ \t]*#define[ \t]+CMAKE_BUILD_TYPE[ \t]+\"HybridRelease\"")

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

# 4. Random123
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")
find_package(Random123 REQUIRED)
if (Random123_FOUND)
  include_directories (${Random123_INCLUDE_DIR})
endif (Random123_FOUND)

# 5. HDF5
find_package(HDF5)
if (HDF5_FOUND)
  set (SKYLARK_HAVE_HDF5
       1
       CACHE
       STRING
       "Enables use of HDF5 Libraries"
       FORCE)
  include_directories (${HDF5_INCLUDE_DIR})
  find_package(ZLIB)
else (HDF5_FOUND)
  set (SKYLARK_HAVE_HDF5
       0
       CACHE
       STRING
       "Enables use of HDF5 Libraries"
       FORCE)
endif (HDF5_FOUND)

#-----------------------------------------------------------------------------
# options

option (USE_FFTW "Build with FFTW support" ON)
if (USE_FFTW)
  set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")
  find_package(FFTW)
  if (FFTW_FOUND)
    include_directories(${FFTW_INCLUDE_DIR})
    link_directories (${FFTW_LIBRARY_DIR})
    set (SKYLARK_HAVE_FFTW
         1
         CACHE
         STRING
         "Enables use of fftw Libraries"
         FORCE)
  endif (FFTW_FOUND)
  find_package(FFTWF)
  if (FFTWF_FOUND)
    include_directories(${FFTWF_INCLUDE_DIR})
    link_directories (${FFTWF_LIBRARY_DIR})
    set (SKYLARK_HAVE_FFTWF
         1
         CACHE
         STRING
         "Enables use of fftw single precision Libraries"
         FORCE)
  endif (FFTWF_FOUND)
endif (USE_FFTW)


option (USE_COMBBLAS "Build with CombBLAS Matrix support" OFF)
if (USE_COMBBLAS)
  set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")
  find_package(CombBLAS REQUIRED)
  if (CombBLAS_FOUND)
    include_directories (${CombBLAS_INCLUDE_DIR})
    link_directories (${CombBLAS_LIBRARY_DIR})

    # use Boost fallback if we don't have c++11 capabilities
    if(NOT CXX11_COMPILER_FLAGS)
      add_definitions(-DCOMBBLAS_BOOST)
    endif(NOT CXX11_COMPILER_FLAGS)

    add_definitions(-D__STDC_LIMIT_MACROS)
    set (SKYLARK_HAVE_COMBBLAS
         1
         CACHE
         STRING
         "Enables use of CombBLAS Libraries"
         FORCE)
  endif (CombBLAS_FOUND)
endif (USE_COMBBLAS)


option (USE_PROFILER "Use Skylark profiling tools" OFF)
if (USE_PROFILER)
  set (SKYLARK_HAVE_PROFILER
       1
       CACHE
       STRING
       "Enables use of profiling extensions"
       FORCE)
endif (USE_PROFILER)


option (USE_HYBRID "Use hybrid MPI/OpenMPI parallelization" ON)
if (USE_HYBRID)
  find_package(OpenMP)
    if (OPENMP_FOUND)
      set (SKYLARK_HAVE_OPENMP
           1
           CACHE
           STRING
           "Enables use of OpenMP extensions"
           FORCE)
      set (CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
      set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    else (OPENMP_FOUND)
      message (FATAL_ERROR "No suitable OpenMP support detected for compiler.")
    endif (OPENMP_FOUND)
endif (USE_HYBRID)


option (BUILD_PYTHON "Whether we should build the python interface" ON)
if (BUILD_PYTHON)

  if (DEFINED ENV{PYTHON_SITE_PACKAGES})
    set (PYTHON_SITE_PACKAGES
      $ENV{PYTHON_SITE_PACKAGES}
    )
    set (PYTHON_SYSTEM_WIDE_INSTALL 0)
  else (DEFINED ENV{PYTHON_SITE_PACKAGES})
    execute_process (
      COMMAND python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(1))"
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set (PYTHON_SYSTEM_WIDE_INSTALL 1)
  endif (DEFINED ENV{PYTHON_SITE_PACKAGES})
  message (STATUS "Installing python modules in: ${PYTHON_SITE_PACKAGES}")

  add_subdirectory(sketch/capi)
  add_subdirectory(python-skylark)
endif (BUILD_PYTHON)

option (BUILD_ML "Whether we should build the machine learning solvers" ON)
if (BUILD_ML)
  add_subdirectory(ml)
endif (BUILD_ML)

option (BUILD_EXAMPLES "Whether we should build the examples" ON)
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif (BUILD_EXAMPLES)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Write out the configure file
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
                ${CMAKE_BINARY_DIR}/config.h)
install (FILES ${CMAKE_BINARY_DIR}/config.h DESTINATION include/skylark/)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Install skylark
#TODO: global recurse is evil (cmake install does not get notified when new
#      files are added). We should compile the header list differently (e.g.
#      install an individual CMakeList files in directories).
file (GLOB_RECURSE HEADER_LIST RELATIVE ${CMAKE_SOURCE_DIR} *.hpp )
foreach (HEADER ${HEADER_LIST})
  string (REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
  install (FILES ${HEADER} DESTINATION include/skylark/${DIR})
endforeach (HEADER)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Uninstall skylark
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/CMake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Unit Tests
enable_testing()
add_subdirectory(tests)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Doxygen
find_package(Doxygen)
if (DOXYGEN_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY
    )
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif (DOXYGEN_FOUND)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Sphinx
find_package(Sphinx)
if (SPHINX_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx/conf.py
        ${CMAKE_CURRENT_BINARY_DIR}/conf.py @ONLY
    )
    add_custom_target(sphinx-doc
        ${SPHINX_EXECUTABLE}
        -q -b html
        -d "${CMAKE_CURRENT_BINARY_DIR}/_cache"
        "${CMAKE_CURRENT_SOURCE_DIR}/doc/sphinx"
        "${CMAKE_CURRENT_BINARY_DIR}/Documentation/sphinx"
        COMMENT "Building HTML documentation with Sphinx"
    )
endif (SPHINX_FOUND)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# configure file
#FIXME: is there a cleaner way to achieve this?
get_directory_property( DirDefs COMPILE_DEFINITIONS )
foreach( d ${DirDefs} )
    set (SKYLARK_DEFS
        ${SKYLARK_DEFS}
        "-D${d}")
endforeach()

configure_file (
  ${CMAKE_CURRENT_SOURCE_DIR}/CMake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake
)

install (
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config_install.cmake
  DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/${PROJECT_NAME}"
  RENAME ${PROJECT_NAME}Config.cmake
)
#----------------------------------------------------------------------------

