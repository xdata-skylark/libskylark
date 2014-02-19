if (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)
  add_executable(hilbert_train train.cpp)
  target_link_libraries(hilbert_train
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES}
                        ${HDF5_LIBRARIES}
                        ${ZLIB_LIBRARIES}
                        ${OpenMP_CXX_FLAGS})
  set_target_properties(hilbert_train
                        PROPERTIES COMPILE_FLAGS
                        "-fopenmp \\
                        -Wall \\
                        -Wno-write-strings \\
                        -Wno-strict-aliasing \\
                        -Wno-format \\
                        -Wno-deprecated \\
                        -Wno-unused-variable \\
                        -Wno-sign-compare")

endif (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)

