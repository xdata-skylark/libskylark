ADD_DEFINITIONS(-DSKYLARK_AVOID_BOOST_PO)

# force static compilation
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
set(CMAKE_EXE_LINKER_FLAGS -static)

set(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS)       # remove -Wl,-Bdynamic
set(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)
set(CMAKE_SHARED_LIBRARY_C_FLAGS)         # remove -fPIC
set(CMAKE_SHARED_LIBRARY_CXX_FLAGS)
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)    # remove -rdynamic
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)

set(Boost_USE_STATIC_LIBS ON)


set(GCC_ROOT  "/bgsys/drivers/ppcfloor/gnu-linux")
set(GCC_NAME  "powerpc64-bgq-linux")
set(CLANG_ROOT "$ENV{CLANG_ROOT}")
set(CLANG_MPI_ROOT "$ENV{CLANG_MPI_ROOT}")
set(IBMCMP_ROOT "$ENV{IBM_MAIN_DIR}")

set(ESSL_LIB "/opt/ibmmath/lib64/")

set(MPI_ROOT   "/bgsys/drivers/ppcfloor/comm/gcc")
set(PAMI_ROOT  "/bgsys/drivers/ppcfloor/comm/sys")
set(SPI_ROOT   "/bgsys/drivers/ppcfloor/spi")

# The serial compilers
set(CMAKE_C_COMPILER   "${CLANG_MPI_ROOT}/bin/mpiclang")
set(CMAKE_CXX_COMPILER "${CLANG_MPI_ROOT}/bin/mpiclang++11")
set(CMAKE_Fortran_COMPILER "${GCC_ROOT}/bin/${GCC_NAME}-gfortran")

# The MPI wrappers for the C and C++ compilers
set(MPI_C_COMPILER   "${CLANG_MPI_ROOT}/bin/mpiclang")
set(MPI_CXX_COMPILER "${CLANG_MPI_ROOT}/bin/mpiclang++11")

set(MPI_C_COMPILE_FLAGS    "")
set(MPI_CXX_COMPILE_FLAGS  "")
set(MPI_C_INCLUDE_PATH     "${MPI_ROOT}/include")
set(MPI_CXX_INCLUDE_PATH   "${MPI_ROOT}/include")
set(MPI_C_LINK_FLAGS       "-L${MPI_ROOT}/lib -L${PAMI_ROOT}/lib -L${SPI_ROOT}/lib")
set(MPI_CXX_LINK_FLAGS     "${MPI_C_LINK_FLAGS}")
set(MPI_C_LIBRARIES        "${MPI_C_LINK_FLAGS}   -lSPI -lSPI_cnk -lrt -lpthread -lstdc++ -lpthread")
set(MPI_CXX_LIBRARIES      "${MPI_CXX_LINK_FLAGS} ${MPI_C_LIBRARIES}")

# set the search path for the environment coming with the compiler
# and a directory where you can install your own compiled software
set(CMAKE_FIND_ROOT_PATH
    /bgsys/drivers/ppcfloor
    /bgsys/drivers/ppcfloor/spi
    ${CLANG_MPI_ROOT}
)

set(XLF_LIB "/opt/ibmcmp/xlf/bg/14.1/bglib64")
set(XLSMP_LIB "/opt/ibmcmp/xlsmp/bg/3.1/bglib64")

if(USE_HYBRID)
  set(MATH_LIBS "-L${ESSL_LIB} -lesslsmpbg -L$ENV{LAPACK_LIB} -llapack -L${ESSL_LIB} -lesslsmpbg -L${XLF_LIB} -lxlf90_r -L${XLSMP_LIB} -lxlsmp -lxlopt -lxlfmath -lxl -lpthread -ldl -Wl,--allow-multiple-definition")
else(USE_HYBRID)
  set(MATH_LIBS "-L${ESSL_LIB} -lesslbg -L$ENV{LAPACK_LIB} -llapack -L${ESSL_LIB} -lesslbg -L${XLF_LIB} -lxlf90_r -L${XLSMP_LIB} -lxlomp_ser -lxlopt -lxlfmath -lxl -lpthread -ldl -Wl,--allow-multiple-definition")
endif(USE_HYBRID)
