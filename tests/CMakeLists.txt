if (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)
  add_executable(sparse_test SparsityTransformTest.cpp)
  target_link_libraries(sparse_test
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Plcg_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
  set_target_properties(sparse_test
                        PROPERTIES COMPILE_FLAGS
                        "-Wall \\
                        -Wno-write-strings \\
                        -Wno-strict-aliasing \\
                        -Wno-format \\
                        -Wno-deprecated \\
                        -Wno-unused-variable \\
                        -Wno-sign-compare")


  add_test( sparse_cwt_test sparse_test )
endif (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)


add_subdirectory(functionality)
