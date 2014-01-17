if (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)

  add_executable(sparse_elemental_apply SparseSketchApplyElementalTest.cpp)
  target_link_libraries(sparse_elemental_apply
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${SKYLARK_LIBS}
                        ${Boost_LIBRARIES})
  set_target_properties(sparse_elemental_apply
                        PROPERTIES COMPILE_FLAGS
                        "-Wall \\
                        -Wno-write-strings \\
                        -Wno-strict-aliasing \\
                        -Wno-format \\
                        -Wno-deprecated \\
                        -Wno-unused-variable \\
                        -Wno-sign-compare")


  add_test( sparse_elemental_apply_test sparse_elemental_apply )

endif (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)

if (SKYLARK_HAVE_COMBBLAS)

  #FIXME: not nice that we need Elemental_LIBRARY here..
  add_executable(sparse_cb_apply SparseSketchApplyCombBLASTest.cpp)
  target_link_libraries( sparse_cb_apply
                         ${SKYLARK_LIBS}
                         ${Elemental_LIBRARY}
                         ${CombBLAS_LIBRARIES}
                         ${Boost_LIBRARIES} )
  add_test( sparse_cb_apply_test sparse_cb_apply )

endif (SKYLARK_HAVE_COMBBLAS)
