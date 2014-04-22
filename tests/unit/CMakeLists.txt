if(SKYLARK_HAVE_ELEMENTAL)
  set(DEP_LIBRARIES ${Elemental_LIBRARY})
endif(SKYLARK_HAVE_ELEMENTAL)

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

  add_executable(sparse_cb_apply SparseSketchApplyCombBLASTest.cpp)
  target_link_libraries( sparse_cb_apply
                         ${SKYLARK_LIBS}
                         ${DEP_LIBRARIES}
                         ${CombBLAS_LIBRARIES}
                         ${Boost_LIBRARIES} )
  add_test( sparse_cb_apply_test sparse_cb_apply )

  add_executable(serialization_test SerializationTest.cpp)
  target_link_libraries( serialization_test
                         ${SKYLARK_LIBS}
                         ${DEP_LIBRARIES}
                         ${CombBLAS_LIBRARIES}
                         ${Boost_LIBRARIES} )
  add_test( serialization_test serialization_test )

endif (SKYLARK_HAVE_COMBBLAS)

add_executable(local_sparse_apply LocalSparseSketchApply.cpp)
target_link_libraries( local_sparse_apply
                       ${SKYLARK_LIBS}
                       ${DEP_LIBRARIES}
                       ${Boost_LIBRARIES} )
add_test( local_sparse_apply_test local_sparse_apply )

if (SKYLARK_HAVE_ELEMENTAL)

  add_executable(svd_elemental_test SVDElementalTest.cpp)
  target_link_libraries(svd_elemental_test
                        ${SKYLARK_LIBS}
                        ${Elemental_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${Boost_LIBRARIES})
  add_test( svd_elemental_test svd_elemental_test )

endif (SKYLARK_HAVE_ELEMENTAL)
