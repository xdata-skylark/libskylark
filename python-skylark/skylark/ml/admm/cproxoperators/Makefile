# THIS REALLY SHOULD BE INTEGRATED INTO CMAKE (!)
PYTHON_INC = /usr/include/python2.7
NUMPY_INC = /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/

_libproxoperators.so: libproxoperators.c
	gcc -I$(PYTHON_INC) -I$(NUMPY_INC) -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -o \
	libproxoperators.o libproxoperators.c
	gcc -fPIC -shared -Wl,-soname=libcproxoperators.so \
	-o "_libproxoperators.so" ./libproxoperators.o -lpython2.7

clean:
	rm *.so *.o
