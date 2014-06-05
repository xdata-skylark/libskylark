Matrices in libskylark
***********************
  
Libskylark is being written to support a variety of local and distributed, dense and sparse 
matrices. Libskylark relies on the `Elemental <http://libelemental.org>`_ library for dense matrices 
and `Combinatorial BLAS <http://gauss.cs.ucsb.edu/~aydin/CombBLAS/html/>`_ for Sparse matrices. 
Elemental provides a variety of 2D and 1D distributed matrix types, while CombBLAS uses 2D block 
cyclic distribution with underlying data structures optimized to exploit sparsity. In addition, 
libskylark also provides support for local sparse matrices, and local numpy arrays. 

