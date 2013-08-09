if (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)
  add_executable(elemental elemental.cpp)
  target_link_libraries(elemental
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
  set_target_properties(elemental
                        PROPERTIES COMPILE_FLAGS
                        "-Wall \\
                        -Wno-write-strings \\
                        -Wno-strict-aliasing \\
                        -Wno-format \\
                        -Wno-deprecated \\
                        -Wno-unused-variable \\
                        -Wno-sign-compare")

  add_executable(regression regression.cpp)
  target_link_libraries(regression
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
  set_target_properties(regression
                        PROPERTIES COMPILE_FLAGS
                        "-Wall \\
                        -Wno-write-strings \\
                        -Wno-strict-aliasing \\
                        -Wno-format \\
                        -Wno-deprecated \\
                        -Wno-unused-variable \\
                        -Wno-sign-compare")

  add_executable(svd svd.cpp)
  target_link_libraries(svd
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
  set_target_properties(svd
                        PROPERTIES COMPILE_FLAGS
                        "-Wall \\
                        -Wno-write-strings \\
                        -Wno-strict-aliasing \\
                        -Wno-format \\
                        -Wno-deprecated \\
                        -Wno-unused-variable \\
                        -Wno-sign-compare")

endif (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)
