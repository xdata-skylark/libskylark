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
