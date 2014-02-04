_libproxoperators.so: libproxoperators.c
	gcc -I/usr/include/python2.7/ -I/usr/include/python2.7/numpy -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -o libproxoperators.o libproxoperators.c
	gcc -fPIC -flat_namespace -shared -Wl,-soname=libcproxoperators.so -o "_libproxoperators.so"  ./libproxoperators.o   -lpython2.7

clean:
	rm *.so *.o
