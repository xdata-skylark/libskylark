option (USE_KISSFFT "Build with KISSFFT support" OFF)
SET(KISSFFT_SOURCE_FILES "")
if (USE_KISSFFT)
  set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules")
  find_package(KISSFFT)
  if (KISSFFT_FOUND)
    include_directories(${KISSFFT_INCLUDE_DIR})

    SET(KISSFFT_SOURCE_FILES
        ${KISSFFT_INCLUDE_DIR}/kissfft.hh
    )

    set (SKYLARK_HAVE_KISSFFT
         1
         CACHE
         STRING
         "Enables use of kissfft Libraries"
         FORCE)
  endif (KISSFFT_FOUND)
endif (USE_KISSFFT)