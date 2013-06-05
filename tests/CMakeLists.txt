add_subdirectory(internal)
add_subdirectory(functionality)
add_subdirectory(performance)

add_custom_target(test
    ctest -E Perf
)

add_custom_target(perf-test
    ctest -R Perf
)

