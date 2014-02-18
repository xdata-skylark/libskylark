if (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)

  add_executable(sparse_elemental_apply SparseSketchApplyElementalTest.cpp)
  target_link_libraries(sparse_elemental_apply
                        ${SKYLARK_LIBS}
                        ${Elemental_LIBRARY}
                        ${FFTW_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${Boost_LIBRARIES})
  add_test( sparse_elemental_apply_test sparse_elemental_apply )

endif (SKYLARK_HAVE_ELEMENTAL AND SKYLARK_HAVE_FFTW)

if (SKYLARK_HAVE_COMBBLAS)

  if(SKYLARK_HAVE_ELEMENTAL)
    set(DEP_LIBRARIES ${Elemental_LIBRARY})
  endif(SKYLARK_HAVE_ELEMENTAL)

  add_executable(sparse_cb_apply SparseSketchApplyCombBLASTest.cpp)
  target_link_libraries( sparse_cb_apply
                         ${SKYLARK_LIBS}
                         ${DEP_LIBRARIES}
                         ${CombBLAS_LIBRARIES}
                         ${Boost_LIBRARIES} )
  add_test( sparse_cb_apply_test sparse_cb_apply )

  add_executable(simple_json_parser SimpleJSONParserTest.cpp)
  target_link_libraries( simple_json_parser
                         ${SKYLARK_LIBS}
                         ${DEP_LIBRARIES}
                         ${CombBLAS_LIBRARIES}
                         ${Boost_LIBRARIES} )
  add_test( simple_json_parser_test simple_json_parser )

endif (SKYLARK_HAVE_COMBBLAS)
