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

if (FALSE)
# Commenting this out for now as we don't have a functioning 
# SVD implementation that uses sketching
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
endif(FALSE)

endif (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)

if (SKYLARK_HAVE_COMBBLAS AND SKYLARK_HAVE_ELEMENTAL)
  add_executable(lsqr lsqr.cpp)
  target_link_libraries(lsqr
                        ${Elemental_LIBRARY}
                        ${CombBLAS_LIBRARIES}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
  set_target_properties(lsqr
                        PROPERTIES COMPILE_FLAGS
                        "-Wall \\
                        -Wno-write-strings \\
                        -Wno-strict-aliasing \\
                        -Wno-format \\
                        -Wno-deprecated \\
                        -Wno-unused-variable \\
                        -Wno-sign-compare")
endif (SKYLARK_HAVE_COMBBLAS AND SKYLARK_HAVE_ELEMENTAL)
