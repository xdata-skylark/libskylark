.. highlight:: rst

Numerical Linear Algebra
**************************

Randomized Least Squares Regression
====================================



Randomized Singular Value Decomposition
========================================

The randomized SVD functionality provides a distributed implementation of algorithms described in
   
	Halko, N. and Martinsson, P.G, and Tropp J., `Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions <http://arxiv.org/abs/0909.4061>`_ , SIAM Rev., Survey and Review section, Vol. 53, num. 2, pp. 217-288, 2011

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
     skylark::nla::rand_svd_params_t params(sketch_size-target_rank);
   
     /* create a rand_svd object parameterized by the Sketch */
     skylark::nla::randsvd_t<skylark::sketch::JLT_t> rand_svd;

     ...
     /* Call the randomized SVD algorithm on Elemental or CombBLAS matrix A */
     rand_svd(A, target_rank, U, S, V, params, context);

The **rand_svd** function accepts certain combinations of matrix types for the input A and the SVD factors: 
U, S and V. For a running example, please see *libskylark/examples/rand_svd.cpp*.


 
