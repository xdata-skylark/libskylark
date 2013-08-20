cmake_minimum_required (VERSION 2.8.2)

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
        RelWithDebInfo MinSizeRel."
        FORCE
      )
endif (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

message (STATUS "Build type is: " ${CMAKE_BUILD_TYPE})

# Select flags.
set (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")
set (CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")

message (STATUS "Building on ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR} on
                 $ENV{HOSTNAME}")

# Add the include directory path for SKYLARK. Remember that only the top dir
# is added for the include, meaning that all the subdirs have to be explictly
# referred. For example,
# #include "../utility/dense_1D.hpp"
# is used in sketch/ subdirectory.
include_directories (${CMAKE_SOURCE_DIR})

#---------------------------------------------------------------------------
# dependent packages
# 1. Find MPI --- we are not adding any include directories here because we
# will be using the mpi compilers, which adds these libraries by default.
find_package (MPI REQUIRED)
set (CMAKE_CXX_COMPILER ${MPI_COMPILER})

# 2. Find Boost with the relevant packages --- Use dynamic boost!
# Without dynamic linking, it's tough to create python bindings.
set(BOOST_ROOT $ENV{BOOST_ROOT})
set(BOOST_NO_SYSTEM_PATHS TRUE)
set(BOOST_INCLUDEDIR $ENV{BOOST_ROOT})
set(BOOST_LIBRARYDIR $ENV{BOOST_ROOT}}/lib)
set(BOOST_MIN_VERSION 1.49.0)
find_package (Boost
              REQUIRED
              mpi
              serialization
              random)
if (Boost_FOUND)
  include_directories (${Boost_INCLUDE_DIRS})
  link_directories (${Boost_LIBRARY_DIRS})
endif (Boost_FOUND)

# 3. BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Set up a dummy variable that lists the common libraries to include.
# Note that Elemental is added to this list only later.
set (SKYLARK_LIBS
    m  # libmath
    ${BLAS_LIBRARIES} # blas
    ${LAPACK_LIBRARIES} # lapack
)

#-----------------------------------------------------------------------------
# options

option (USE_ELEMENTAL "Build with Elemental Matrix support" ON)
if (USE_ELEMENTAL)
  set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")
  find_package(Elemental REQUIRED)
  if (Elemental_FOUND)
    include_directories (${Elemental_INCLUDE_DIR})
    link_directories (${Elemental_LIBRARY_DIR})
    set (SKYLARK_HAVE_ELEMENTAL
         1
         CACHE
         STRING
         "Enables use of Elemental Libraries"
         FORCE)
  endif (Elemental_FOUND)
endif (USE_ELEMENTAL)


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
endif (USE_FFTW)


option (USE_COMBBLAS "Build with CombBLAS Matrix support" OFF)
if (USE_COMBBLAS)
  set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")
  find_package(CombBLAS REQUIRED)
  if (CombBLAS_FOUND)
    include_directories (${CombBLAS_INCLUDE_DIR})
    link_directories (${CombBLAS_LIBRARY_DIR})
    #FIXME: Boost only necessary if compiler is not c++11 compatible
    add_definitions(-DCOMBBLAS_BOOST)
    add_definitions(-D__STDC_LIMIT_MACROS)
    set (SKYLARK_HAVE_COMBBLAS
         1
         CACHE
         STRING
         "Enables use of CombBLAS Libraries"
         FORCE)
  endif (CombBLAS_FOUND)
endif (USE_COMBBLAS)


option (BUILD_PYTHON "Whether we should build the python interface" ON)
if (BUILD_PYTHON)

  if (DEFINED ENV{PYTHON_SITE_PACKAGES})
    set (PYTHON_SITE_PACKAGES
      $ENV{PYTHON_SITE_PACKAGES}
    )
    set (PYTHON_SYSTEM_WIDE_INSTALL 0)
  else (DEFINED ENV{PYTHON_SITE_PACKAGES})
    execute_process (
      COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set (PYTHON_SYSTEM_WIDE_INSTALL 1)
  endif (DEFINED ENV{PYTHON_SITE_PACKAGES})
  message (STATUS "Installing python modules in: ${PYTHON_SITE_PACKAGES}")

  add_subdirectory(sketch/capi)
  add_subdirectory(python-skylark)
endif (BUILD_PYTHON)

option (BUILD_EXAMPLES "Whether we should build the examples" ON)
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif (BUILD_EXAMPLES)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Write out the configure file
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
                ${CMAKE_CURRENT_SOURCE_DIR}/config.h)
install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/config.h DESTINATION include/skylark/)
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
        ${CMAKE_CURRENT_SOURCE_DIR}/python-skylark/doc/source/conf.py
        ${CMAKE_CURRENT_BINARY_DIR}/conf.py @ONLY
    )
    add_custom_target(sphinx-doc
        ${SPHINX_EXECUTABLE}
        -q -b html
        -d "${CMAKE_CURRENT_BINARY_DIR}/_cache"
        "${CMAKE_CURRENT_SOURCE_DIR}/python-skylark/doc/source"
        "${CMAKE_CURRENT_BINARY_DIR}/Documentation/sphinx"
        COMMENT "Building HTML documentation with Sphinx"
    )
endif (SPHINX_FOUND)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# configure file
get_directory_property( DirDefs COMPILE_DEFINITIONS )
foreach( d ${DirDefs} )
    set (CMAKE_DEFS
        ${CMAKE_DEFS}
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

