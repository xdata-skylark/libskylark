find_package(PythonLibs REQUIRED)

include_directories (
    ${CMAKE_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIRS}
)

set(SKYLARK_SKETCH_SRCS
    module.cpp
    context.cpp
    JLT.cpp
    transforms.cpp
    documentation.cpp
)

add_library(sketch
            SHARED
            ${SKYLARK_SKETCH_SRCS}
)
target_link_libraries(sketch
                      ${SKYLARK_LIBS}
                      ${Boost_LIBRARIES}
                      ${PYTHON_LIBRARIES})
set_target_properties(sketch
                      PROPERTIES COMPILE_FLAGS
                      "-Wall \\
                      -Werror \\
                      -Wno-strict-aliasing \\
                      -Wno-sign-compare \\
                      -Wno-comment \\
                      -Wno-unused-variable \\
                      -fPIC")
set_target_properties(sketch PROPERTIES PREFIX "_")
#TODO: why do we have to have *.so file ending (at least on Mac OSX)
set_target_properties(sketch PROPERTIES SUFFIX ".so")

INSTALL (
    TARGETS sketch
    DESTINATION ${PYTHON_SITE_PACKAGES}/skylark/sketch
)

# and install all Python resources
install (
  CODE "file(GLOB _py_resources \"${CMAKE_CURRENT_SOURCE_DIR}/resources/*\" )"
  CODE "file(INSTALL \${_py_resources} DESTINATION \"${PYTHON_SITE_PACKAGES}/skylark/sketch\" )"
)

