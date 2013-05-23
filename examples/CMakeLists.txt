if (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)
  add_executable(elemental elemental.cpp)
  target_link_libraries(elemental
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Plcg_LIBRARY}
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
endif (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)
