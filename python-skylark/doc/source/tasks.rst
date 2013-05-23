Task Plans
============

Task 1.1: *Design and Concept Document*
---------------------------------------
For our first version (v0.0.1  - May 2013), we will target a shared memory implementation, where some primitives 
may be multi-threaded while others may only run in single-threaded mode. The goal is to have an efficient piece of software
that you can run on a commodity multicore laptop, both for practical applications as well as for research prototyping. 
In the meantime, we will also develop a strategy for fully distributed implementations, 
communication-avoiding versions of the algorithms and so on.

The following functionality is being planned for v0.0.1:
 
Sketching Transforms
######################

#. Random Sampling
#. JLT with Gaussian and Sign Matrices 
#. FJLT (FFT Based) 
#. Sparse JL/Count-Min Sketch/Hashing-based 
  
*Parallelization Strategy* 

* Random Sampling: 
 
	Variations we want:  
	- sample with replacement given a probability vector over rows or columns, or both (do we need without replacement too?) 
	- bernoulli sampling
	  
	Materialize a selection bit-vector - broadcast to all tasks which do the local sampling.

* JLT with Gaussian and Sign Matrices
 
	Lets assume BxB blocks for now?  Parallel Pseudo-random Number Generators. Use Anju's cache-optimal matrix multiplication code.

* FJLT/FFT-based: Haim's blendenpik implementation. 

	Use FFTW's multithreading (other operations can be parallelized with openMP)?


Least Squares Library
#######################

* Standard Direct and Iterative Solvers (CGLS/LSQR) : Use scipy/numpy. For iterative solvers, may need to look in. 

* Blendenpik : Haim
Need a python version

* LSRN : Mike Mahoney
Simply expose already implemented python code.


Leverage Score Computations
##############################
Haim/Christos to provide details. 

Low-Rank Approximations
##########################
Christos to provide details. 


Principal Component Analysis
#############################

Christos to provide details.

Canonical Correlation Analysis
################################

Christos to provide details

Nonlinear Prediction
#####################

Rahimi-Recht needs Gaussian Sketching, followed by cosine transformation, followed by least squares solver. 

Leverage score computations can be used for cross-validation, outlier analysis.


Matrix Completion and Missing-data Imputation
##############################################

Haim/Vikas to work out details.

Robust Regression
######################

This is not in our initial timeline, but I think we should try to do it (Vikas).

Task 1.1: TimeLine
-------------------

Lets fill out in January.


Task 1.2: *Machine Learning and Big Data Applications*
------------------------------------------------------

TBD.
