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

find_package(PythonInterp REQUIRED)
message (STATUS "Using Python interpreter to run tests: {PYTHON_EXECUTABLE}")
#FIXME: Because of mpi4py/kdt MPI initialization issue we cannot run the test
#       in hybrid mode when we need to call MPI.Init_thread().
if(!USE_HYBRID)
    add_test(NAME ${NAME}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_test.py ${CMAKE_SOURCE_DIR}
    )
endif(!USE_HYBRID)

# run all python tests
#file (GLOB PY_TEST_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py )
#foreach (TEST ${PY_TEST_LIST})
    #string (REGEX REPLACE "([a-zA-Z]+).py" "\\1" NAME ${TEST})

    #add_test(NAME ${NAME}
        #COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST} ${CMAKE_SOURCE_DIR}
    #)
#endforeach (TEST)

if (SKYLARK_HAVE_ELEMENTAL)

  add_executable(svd_elemental_test SVDElementalTest.cpp)
  target_link_libraries(svd_elemental_test
                        ${SKYLARK_LIBS}
                        ${Elemental_LIBRARY}
                        ${Pmrrr_LIBRARY}
                        ${Boost_LIBRARIES})
  add_test( svd_elemental_test svd_elemental_test )

endif (SKYLARK_HAVE_ELEMENTAL)
