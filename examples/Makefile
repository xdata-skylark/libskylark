SOFTWARE_HOME=${HOME}/Software

MPICXX  = mpic++
MPIRUN  = mpirun
FORTRAN = gfortran

ifeq ($(shell uname -s),Darwin)
  EXTRAFLAG=-framework vecLib 
endif
ifeq ($(shell uname -s),Linux)
  EXTRALIB=-llapack -lblas -lgfortran
endif

EXTRALIB:=$(EXTRALIB) \
  -L${SOFTWARE_HOME}/lib \
  -lboost_mpi \
  -lboost_mpi_python \
  -lboost_serialization \
  -lboost_python \
  -lelemental

DBGFLAG=-g -O0
OPTIFLAG=-O3

INCLIBS = -lm $(EXTRALIB)
INCFLAGS= -I${SOFTWARE_HOME}/include \
          -I.. 

CXXFLAGS:=$(CXXFLAGS) \
          -Wno-write-strings \
          -Wno-format \
          -Wno-deprecated \
          $(EXTRAFLAG)

# Need to add the relevant headers for dependency
HEADERS = 

all: dense-1D elemental

elemental: elemental.cpp $(HEADERS) 
	${MPICXX} ${CXXFLAGS} ${OPTIFLAG} ${INCFLAGS} $< -o $@ ${INCLIBS}

dense-1D: dense-1D.cpp $(HEADERS) 
	${MPICXX} ${CXXFLAGS} ${OPTIFLAG} ${INCFLAGS} $< -o $@ ${INCLIBS}

.PHONY: clean

clean:
	rm -rf *.o dense-1D dense-1D-dbg elemental elemental-dbg *.dSYM out error
