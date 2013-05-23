include /usr/local/conf/elemvariables

BASEDIR = /home/haimav/Coding/xdata.git/skylark

MPIRUN  = mpirun
FORTRAN = gfortran

ifeq ($(shell uname -s),Darwin)
  EXTRAFLAG=-framework vecLib 
endif
ifeq ($(shell uname -s),Linux)
  EXTRALIB=/usr/lib/liblapack.so /usr/lib/libblas.so -lgfortran -lfftw3 -lm
endif

EXTRALIB:=$(EXTRALIB) \
  -L${SOFTWARE_HOME}/lib \
  -lboost_mpi \
  -lboost_mpi_python \
  -lboost_serialization \
  -lboost_python

DBGFLAG=-g -O0
OPTIFLAG=-O3

INCLIBS = -lm $(EXTRALIB)
INCFLAGS= -I $(BASEDIR) -I${ELEM_INC} ${MPI_CXX_INCLUDE_STRING}
CXXFLAGS:=$(CXXFLAGS) -fPIC -Wno-write-strings -Wno-format -Wno-deprecated $(EXTRAFLAG)

HEADERS = $(BASEDIR)/skylark.hpp


all: examples/elemental

examples/elemental: examples/elemental.cpp $(HEADERS) 
	${CXX} ${CXXFLAGS} ${OPTIFLAG} ${INCFLAGS} $< -o $@ ${ELEM_LINK_FLAGS} ${ELEM_LIBS} ${EXTRALIB}

examples/elemental.dbg: examples/elemental.cpp $(HEADERS) 
	${CXX} ${CXXFLAGS} ${DBGFLAG} ${INCFLAGS} $< -o $@ ${ELEM_LINK_FLAGS} ${ELEM_LIBS} ${EXTRALIB}

sketch/ctypes_python/libpyskylark.so: sketch/capi/capi.cpp
	${CXX} ${CXXFLAGS} ${OPTIFLAG} ${INCFLAGS} -c -fPIC sketch/capi/capi.cpp -o sketch/capi/capi.o
	${CXX} -shared -Wl,-soname,libpyskylark.so -o $@ sketch/capi/capi.o  ${ELEM_LINK_FLAGS} ${ELEM_LIBS} ${EXTRALIB}

.PHONY: clean

clean:
	rm -rf *.o examples/elemental examples/elemental.dbg *.dSYM out error
