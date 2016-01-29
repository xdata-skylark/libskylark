option (USE_SPIRALWHT "Build with SpiralWHT support" OFF)
if (USE_SPIRALWHT)
  if (SPIRALWHT_FOUND)
    include_directories(${SPIRALWHT_INCLUDE_DIR})
    link_directories (${SPIRALWHT_LIBRARY_DIR})
    set (SKYLARK_HAVE_SPIRALWHT
         1
         CACHE
         STRING
         "Enables use of spiralwht Libraries"
         FORCE)
  endif (SPIRALWHT_FOUND)
endif (USE_SPIRALWHT)
