include_directories (
    ${CMAKE_SOURCE_DIR}
)

set(SKYLARK_CAPI_SRCS
    capi.cpp
)

add_library(cskylark
            SHARED
            ${SKYLARK_CAPI_SRCS}
)
target_link_libraries(cskylark
                      ${SKYLARK_LIBS}
                      ${FFTW_LIBRARY}
                      ${Elemental_LIBRARY}
                      ${Boost_LIBRARIES}
)
set_target_properties(cskylark
                      PROPERTIES COMPILE_FLAGS
                      "-Wall \\
                      -Werror \\
                      -Wno-strict-aliasing \\
                      -Wno-sign-compare \\
                      -Wno-comment \\
                      -Wno-unused-variable \\
                      -fPIC"
)
set_target_properties(cskylark PROPERTIES SUFFIX ".so")

install (
    TARGETS cskylark
    DESTINATION lib/
)

install (
    FILES "sketchc.hpp"
    DESTINATION include/skylark/
)

