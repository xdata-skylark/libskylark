find_package(PythonInterp REQUIRED)
message (STATUS "Using Python interpreter to run tests: ${PYTHON_EXECUTABLE}")


file (GLOB PY_TEST_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *_performance.py )
foreach (TEST ${PY_TEST_LIST})
    string (REGEX REPLACE "([a-zA-Z]+)_performance.py" "\\1" NAME ${TEST})

    #FIXME: fix number of procs range with CMAKE config
    foreach(val RANGE 1 16 1)
        add_test(NAME Perf${NAME}_np${val}
            COMMAND mpirun -np ${val} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST}
        )
    endforeach (val)
endforeach (TEST)

# generate plots
if (DEFINED ENV{PERF_INSTALL_DIR})
  add_custom_target(perf-plots
    ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/generate_plot.py
    ${CMAKE_BINARY_DIR}/tests/performance $ENV{PERF_INSTALL_DIR}
  )
else (DEFINED ENV{PERF_INSTALL_DIR})
  message(STATUS "No target directory for performance plots specified!  Export PERF_INSTALL_DIR!")
endif (DEFINED ENV{PERF_INSTALL_DIR})
