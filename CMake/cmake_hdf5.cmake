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
