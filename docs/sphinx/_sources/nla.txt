.. highlight:: rst

Numerical Linear Algebra
**************************

Randomized Least Squares Regression
====================================

The randomized regression functionality provides distributed implementation of algorithms described in 
	
	* Avron, H., Maymounkov, P. and Toledo, S., `Supercharging LAPACK's Least Squares Solver <http://dl.acm.org/citation.cfm?id=1958633>`_ , 2010
	* Meng, X., Saunders, M.A. and Mahoney, M. W, `LSRN: A Paralllel Iterative Solver for Strongly Over- or Under-Determined Systems <http://arxiv.org/abs/1109.5981>`_ , 2012

A flavor of usage is given in the code snippet below. The usage mirrors Elemental's `LeastSquares <http://libelemental.org/documentation/0.83/lapack-like/solve.html>`_ function, but the solve is accelerated using sketching (specified internally). 

.. code-block:: cpp

     #include <elemental.hpp>
     #include <skylark.hpp>
     ...
     // Setup regression problem with coefficient matrix A and target matrix B
     ...
     
     skybase::context_t context(23234);

     // Solve the Least Squres problem of minimizing || AX - B||_2 over X
     skylark::nla::FastLeastSquares(elem::NORMAL, A, X, B, context);

Please see *libskylark/examples/least_squares.cpp* for a running example.

Randomized Singular Value Decomposition
========================================

The randomized SVD functionality provides a distributed implementation of algorithms described in
    
	* Halko, N. and Martinsson, P.G, and Tropp J., `Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions <http://arxiv.org/abs/0909.4061>`_ , SIAM Rev., Survey and Review section, Vol. 53, num. 2, pp. 217-288, 2011

The prototypical algorithm involves the following steps, given a matrix **A**
	* Compute an approximate orthonormal basis for the range of **A**, as specified by the columns of an orthonormal matrix **Q**.
        * Use **Q** to compute a standard factorization of **A**    

The first step is accelerated using sketching.

A flavor of usage is given in the code snippet below. 

.. code-block:: cpp

     #include <elemental.hpp>
     #include <skylark.hpp>
     ...    
     /* params structure contains parameters of the randomized SVD algorithm */
     /* These parameters include oversampling, number of power iterations and 
     whether repeated QR iterations should be skipped */ 
     
     skylark::nla::rand_svd_params_t params(oversampling_parameter);
   
     /* create a rand_svd object parameterized by the Sketch */
     skylark::nla::randsvd_t<skylark::sketch::JLT_t> rand_svd;

     ...
     /* Call the randomized SVD algorithm on Elemental or CombBLAS matrix A */
     rand_svd(A, target_rank, U, S, V, params, context);

The **rand_svd** function accepts certain combinations of matrix types for the input A and the SVD factors: 
U, S and V. Currently, the matrix types are Elemental MC/MR or elem::Matrix types. 

For a running example, please see *libskylark/examples/rand_svd.cpp*.


 
