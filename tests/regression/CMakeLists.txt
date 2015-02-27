find_package(PythonInterp REQUIRED)
message (STATUS "Using Python interpreter to run tests: ${PYTHON_EXECUTABLE}")


file (GLOB PY_TEST_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py )
foreach (TEST ${PY_TEST_LIST})
    string (REGEX REPLACE "([a-zA-Z]+).py" "\\1" NAME ${TEST})

    add_test(NAME ${NAME}
        COMMAND mpirun -np 1 ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST}
    )
endforeach (TEST)
