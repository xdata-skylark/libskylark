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
