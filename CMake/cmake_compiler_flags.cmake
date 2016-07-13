message(STATUS "Building with ${CMAKE_CXX_COMPILER_ID} compiler")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

    set (SKYLARK_LIBS
        m
    )

    set (CXX11_COMPILER_FLAGS "-std=c++11")
    set (COMPILER_SPEC_FLAGS
        "-W -Wall -Wno-write-strings -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare -Wno-unused-local-typedef"
    )

    set (CMAKE_LIB_LINKER_FLAGS  "${CMAKE_LIB_LINKER_FLAGS} -fPIC")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

    set (SKYLARK_LIBS
        m
    )

    set (CXX11_COMPILER_FLAGS "-std=c++11")
    set (COMPILER_SPEC_FLAGS
        "-W -Wall -Wno-write-strings -Wno-strict-aliasing -Wno-format -Wno-deprecated -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare -Wno-unused-local-typedefs"
    )

    set (CMAKE_LIB_LINKER_FLAGS  "${CMAKE_LIB_LINKER_FLAGS} -fPIC")


elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    #TODO: using Intel C++

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")

    set (COMPILER_SPEC_FLAGS
        "-qtune=qp -qarch=qp -qmaxmem=-1 -qcpluscmt -qstrict"
    )

    include_directories (
    )

    link_directories (
        /opt/ibmcmp/xlf/bg/14.1/bglib64/
        /bgsys/drivers/ppcfloor/spi/lib
        /bgsys/drivers/ppcfloor/bgpm/lib/
    )

    set (SKYLARK_LIBS
        xlf90_r
        xlfmath
    )

    set(BGQ_PROFILING_LIBRARIES
        mpihpm
        mpitrace
        bgpm
    )

else()
    message (FATAL "Unsupported compiler!")

endif()

message (STATUS "CXX11_COMPILER_FLAGS=${CXX11_COMPILER_FLAGS}")
set (CMAKE_CXX_FLAGS "${CXX11_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
