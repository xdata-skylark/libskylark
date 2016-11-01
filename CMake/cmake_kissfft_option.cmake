option (USE_KISSFFT "Build with KISSFFT support" OFF)
if (USE_KISSFFT)
  set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")
  find_package(KISSFFT)
  if (KISSFFT_FOUND)
    include_directories(${KISSFFT_INCLUDE_DIR})
    link_directories (${KISSFFT_LIBRARY_DIR})
    set (SKYLARK_HAVE_KISSFFT
         1
         CACHE
         STRING
         "Enables use of kissfft Libraries"
         FORCE)
  endif (KISSFFT_FOUND)
endif (USE_KISSFFT)