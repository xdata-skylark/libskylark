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

endif (SKYLARK_HAVE_COMBBLAS)


# run all python tests
find_package(PythonInterp REQUIRED)
message (STATUS "Using Python interpreter to run tests: {PYTHON_EXECUTABLE}")
file (GLOB PY_TEST_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py )
foreach (TEST ${PY_TEST_LIST})
    string (REGEX REPLACE "([a-zA-Z]+).py" "\\1" NAME ${TEST})

    add_test(NAME ${NAME}
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST}
    )
endforeach (TEST)
