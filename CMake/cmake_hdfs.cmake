find_package(LIBHDFS)
if (LIBHDFS_FOUND)
  find_package(JNI)
  if (JNI_FOUND)
    set (SKYLARK_HAVE_LIBHDFS
      1
      CACHE
      STRING
      "Enables use of LIBHDFS Libraries"
      FORCE)
    include_directories (${LIBHDFS_INCLUDE_DIR})
  else(JNI_FOUND)
    set (SKYLARK_HAVE_LIBHDFS
      0
      CACHE
      STRING
      "Enables use of LIBHDFS Libraries"
      FORCE)
    IF (NOT HDF5_FIND_QUIETLY)
      MESSAGE(WARNING "LIBHDFS not used because JNI not found!")
    ENDIF (NOT HDF5_FIND_QUIETLY)
  endif (JNI_FOUND)
else (LIBHDFS_FOUND)
  set (SKYLARK_HAVE_LIBHDFS
       0
       CACHE
       STRING
       "Enables use of LIBHDFS Libraries"
       FORCE)
endif (LIBHDFS_FOUND)
