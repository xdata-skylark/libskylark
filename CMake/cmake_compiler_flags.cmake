message(STATUS "Building with ${CMAKE_CXX_COMPILER_ID} compiler")
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

    if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )
        include_directories (
            /usr/llvm-gcc-4.2/lib/gcc/i686-apple-darwin11/4.2.1/include
        )
    endif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin" )

    set (SKYLARK_LIBS
        m
    )

    set (COMPILER_SPEC_FLAGS
        "-W -Wall -Wno-write-strings -Wno-strict-aliasing -Wno-format -Wno-deprecated -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare"
    )

    #set (LINK_FLAGS
    #)

    set (CMAKE_LIB_LINKER_FLAGS  "${CMAKE_LIB_LINKER_FLAGS} -fPIC")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

    set (SKYLARK_LIBS
        m
    )

    set (COMPILER_SPEC_FLAGS
        "-W -Wall -Wno-write-strings -Wno-strict-aliasing -Wno-format -Wno-deprecated -Wno-unused-variable -Wno-unused-parameter -Wno-sign-compare -Wno-unused-local-typedefs"
    )

    #set (LINK_FLAGS
    #)

    set (CMAKE_LIB_LINKER_FLAGS  "${CMAKE_LIB_LINKER_FLAGS} -fPIC")


elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    #TODO: using Intel C++

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")

    set (COMPILER_SPEC_FLAGS
        "-qtune=qp -qarch=qp -qmaxmem=-1 -qcpluscmt -qstrict"
    )
    # -qlanglvl=extended0x

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
