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
