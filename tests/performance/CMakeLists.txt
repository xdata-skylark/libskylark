find_package(PythonInterp REQUIRED)
message (STATUS "Using Python interpreter to run tests: ${PYTHON_EXECUTABLE}")

if (DEFINED ENV{PERF_INSTALL_DIR})
  add_custom_target(perf-test
    mpirun -np 1 ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/helper/master.py
    4 2 True ${CMAKE_CURRENT_SOURCE_DIR} $ENV{PERF_INSTALL_DIR} $ENV{PERF_INSTALL_DIR}/data ${CMAKE_CURRENT_SOURCE_DIR}/remotes.json
  )
else (DEFINED ENV{PERF_INSTALL_DIR})
  message(STATUS "No target directory for performance plots specified!  Export PERF_INSTALL_DIR!")
endif (DEFINED ENV{PERF_INSTALL_DIR})
