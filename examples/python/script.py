from setup import *

" 1. Print out some stuff "
print 'Hello, this is {0} of {1}'.format(mpi.rank, mpi.size)

" 2. Create an array to hold the partition information and partition "
M = 10; S = 5; N = 5;
parts = numpy.zeros((1,(mpi.size+1)), dtype=numpy.int32)
skylark.utility.int32_partitioner.divide (0, 
                                          M, 
                                          mpi.size, 
                                          skylark.utility.int_iterator(parts))
int_map = skylark.utility.mapper(skylark.utility.int_iterator(parts), 
                                 parts.size)

" 3. Create local array for this "
sl_context = skylark.sketch.context(0, mpi.world)
A = skylark.utility.dense_1D_row ('A', 
                                   M, 
                                   N, 
                                   int_map, 
                                   sl_context,
  skylark.utility.dbl_iterator(numpy.random.rand(int_map.range(mpi.rank), N)))

" 4. Create and execute sketch "
JL_sketch = skylark.sketch.JLT(S, M, sl_context)
S = JL_sketch.preview (A, skylark.sketch.left())
S.set_buffer (skylark.utility.dbl_iterator(numpy.zeros((S.range(), S.N))))

" 5. apply "
JL_sketch.apply (A, S, skylark.sketch.left());

" 6. print "
S.pretty_print()
