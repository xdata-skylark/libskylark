find_package(PythonLibs REQUIRED)

include_directories (
    ${CMAKE_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIRS}
)

set(SKYLARK_UTILITY_SRCS
    module.cpp
    partitioner.cpp
    py_iterator.cpp
    mapper.cpp
    dense_1D.cpp
    matrix.cpp
    documentation.cpp
)

add_library(utility
            SHARED
            ${SKYLARK_UTILITY_SRCS}
)
target_link_libraries(utility
                      ${SKYLARK_LIBS}
                      ${Boost_LIBRARIES}
                      ${PYTHON_LIBRARIES})
set_target_properties(utility
                      PROPERTIES COMPILE_FLAGS
                      "-Wall \\
                      -Werror \\
                      -Wno-strict-aliasing \\
                      -Wno-sign-compare \\
                      -Wno-comment \\
                      -Wno-unused-variable \\
                      -fPIC")
set_target_properties(utility PROPERTIES PREFIX "_")
#TODO: why do we have to have *.so file ending (at least on Mac OSX)
set_target_properties(utility PROPERTIES SUFFIX ".so")

install (
    TARGETS utility
    DESTINATION ${PYTHON_SITE_PACKAGES}/skylark/utility
)

# and install all Python resources
install (
  CODE "file(GLOB _py_resources \"${CMAKE_CURRENT_SOURCE_DIR}/resources/*\" )"
  CODE "file(INSTALL \${_py_resources} DESTINATION \"${PYTHON_SITE_PACKAGES}/skylark/utility\" )"
)

