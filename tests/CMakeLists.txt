include(CTest)
set(CTEST_MEMORYCHECK_COMMAND "/usr/bin/valgrind")
set(MEMORYCHECK_COMMAND_OPTIONS "--xml=yes --xml-file=memtest.xml")

add_subdirectory(unit)
add_subdirectory(regression)
add_subdirectory(performance)

add_custom_target(test
    ctest -E Perf
)

add_custom_target(perf-test
    ctest -R Perf
)

add_custom_target(mem-test
    ctest -R Mem -D ExperimentalMemCheck
)

