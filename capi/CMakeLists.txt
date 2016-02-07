# FIXME: This should be removed when the errors in CombBLAS are fixed!
# (1)
# CombBLAS_beta_13_0/psort-1.0/driver/MersenneTwister.h: In member function
# ‘void MTRand::seed(MTRand::uint32*, MTRand::uint32)’:
# CombBLAS_beta_13_0/psort-1.0/driver/MersenneTwister.h:234:42: error: enumeral
# and non-enumeral type in conditional expression [-Werror]
#
# (2)
# CombBLAS_beta_13_0/FullyDistVec.cpp:466:8: error: comparison of unsigned
# expression < 0 is always false [-Werror=type-limits]
#
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/capi.cpp
                             PROPERTIES COMPILE_FLAGS "-Wno-error")

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
                      ${OPTIONAL_LIBS}
                      ${Elemental_LIBRARY}
                      ${Boost_LIBRARIES}
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

