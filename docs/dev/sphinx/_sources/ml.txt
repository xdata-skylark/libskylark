Machine Learning
*****************

Randomized Kernel Methods
==========================
 
libSkylark provides distributed implementations of kernel-based nonlinear models for 
 
	* Regularized Least Squares Regression and Classification
	* Regularized Robust Regression (Least Absolute Deviation loss)
	* Support Vector Machines
        * Multinomial Logistic Regression (classes > 2). 

The following kernels are supported:
	
	* Gaussian, Laplacian and Matern Kernels via Random Fourier Transform (`Rahimi and Recht, 2007 <http://www.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf>`_)
	* Gaussian and Matern Kernels via Fast Random Fourier Transform (`Le, Sarlos and Smola, 2013 <http://jmlr.org/proceedings/papers/v28/le13.html>`_)
	* Gaussian and Matern Kernels via Quasi Random Fourier Transform (`Yang et al, 2014 <http://jmlr.org/proceedings/papers/v32/yangb14.pdf>`_)
	* Polynomial Kernels via Tensor Sketch (`Pahm and Pagh, 2013 <http://www.itu.dk/people/ndap/TensorSketch.pdf>`_) 
	* Exponential Semigroup Kernels via Random Laplace Transform (`Yang et al, 2014 <http://vikas.sindhwani.org/RandomLaplace.pdf>`_)

The implementations combine two ideas:
	* Constructing randomized approximations to Kernel functions *on the fly*
        * Using a distributed optimization solver based on Alternating Directions Method of Multipliers (ADMM)
 
The distributed optimization approach is based on a block-splitting variant of ADMM proposed in `Parikh and Boyd, 2014 <http://web.stanford.edu/~boyd/papers/block_splitting.html>`_
 
The full implementation (under ``libskylark/ml``) is described in the following paper:
	* Sindhwani V. and Avron H., High-performance Kernel Machines with Implicit Distributed Optimization and Randomization, 2014

Standalone Usage 
----------------- 

Building libSkylark creates an executable called **skylark_ml** under CMAKE_PREFIX_INSTALL/bin. This executable can be 
used out-of-the-box for large-scale applications involving kernel-based modeling.
 
.. _ml_io:

Input Data Format
------------------
The implementation supports `LIBSVM <http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/>`_ file format, where 
feature vectors and labels are specified as

::

	0 1:0.2 4:0.5 10:0.3
        5 3:0.3 6:0.1 

Each line begins with a label and followed by index (starting with 1)-value pairs describing the feature vector in 
sparse format. 
  
We also support `HDF5 <http://www.hdfgroup.org/HDF5/>`_ data files. 

	* Dense training data can be described using HDF5 files containing two `HDF5 datasets <http://www.hdfgroup.org/HDF5/Tutor/crtdat.html>`_: 
		* *X* -- n x d matrix  (examples-by-features)
		* *Y* -- n x 1 matrix of labels. 
	* Sparse training data can be described using HDF5 files containing five `HDF5 datasets <http://www.hdfgroup.org/HDF5/Tutor/crtdat.html>`_ specifying a Compressed Row Storage Sparse matrix: 
		* *dimensions*: 3 x 1 matrix [number of features, number of examples, number of nonzeros (nnz)]
                * *indices*: nnz x 1 matrix column indices of non-zero values for CRS datastructure representing the examples-by-features sparse matrix
		* *values*: nnz x 1 non-zero values corresponding to indices
 		* *indptr*: (n+1) x 1 - pointer into indices, values specifying rows
		* *Y*: n x 1 matrix of labels


Examples of such files can be downloaded from `here <http://vikas.sindhwani.org/data.tar.gz>`_. The HDF5 files can be viewed using `HDFView <http://http://www.hdfgroup.org/HDF5/Tutor/hdfview.html>`_. A screenshot is shown below.
 
.. image:: files/hdfview_screenshot.png
    :width: 750 px
    :align: center



Example and Commandline Usage
-----------------------------

Please see :ref:`ml_example`

::

   Usage: skylark_ml [options] --trainfile trainfile --modelfile modelfile
   Usage: skylark_ml --modelfile modelfile --testfile testfile :
     -h [ --help ]                         produce a help message
     -l [ --lossfunction ] arg (=0)        Loss function (0:SQUARED, 1:LAD, 
                                           2:HINGE, 3:LOGISTIC)
     -r [ --regularizer ] arg (=0)         Regularizer (0:L2, 1:L1)
     -k [ --kernel ] arg (=0)              Kernel (0:LINEAR, 1:GAUSSIAN, 
                                           2:POLYNOMIAL, 3:LAPLACIAN, 
                                           4:EXPSEMIGROUP, 5:MATERN)
     -g [ --kernelparam ] arg (=1)         Kernel Parameter
     -x [ --kernelparam2 ] arg (=0)        If Applicable - Second Kernel Parameter
                                           (Polynomial Kernel: c)
     -y [ --kernelparam3 ] arg (=1)        If Applicable - Third Kernel Parameter 
                                           (Polynomial Kernel: gamma)
     -c [ --lambda ] arg (=0)              Regularization Parameter
     -e [ --tolerance ] arg (=0.001)       Tolerance
     --rho arg (=1)                        ADMM rho parameter
     -s [ --seed ] arg (=12345)            Seed for Random Number Generator
     -f [ --randomfeatures ] arg (=100)    Number of Random Features (default: 
                                           100)
     -n [ --numfeaturepartitions ] arg (=1)
                                           Number of Feature Partitions (default: 
                                           1)
     -t [ --numthreads ] arg (=1)          Number of Threads (default: 1)
     --regression                          Build a regression model(default is 
                                           classification).
     --usefast                             Use 'fast' feature mapping, if 
                                           available. Default is to use 'regular' 
                                           mapping.
     -q [ --usequasi ] arg (=0)            If possible, change the underlying 
                                           sequence of samples (0:Regular/Monte 
                                           Carlo, 1:Leaped Halton)
     --cachetransforms                     Cache feature expanded data (faster, 
                                           but more memory demanding).
     --fileformat arg (=0)                 Fileformat (default: 0 (libsvm->dense),
                                           1 (libsvm->sparse), 2 (hdf5->dense), 3 
                                           (hdf5->sparse)
     -i [ --MAXITER ] arg (=20)            Maximum Number of Iterations (default: 
                                           10)
     --trainfile arg                       Training data file (required in 
                                           training mode)
     --modelfile arg                       Model output file
     --valfile arg                         Validation file (optional)
     --testfile arg                        Test file (required in testing mode)
     --outputfile arg                      Base name for output file (will attach 
                                           .txt suffix)

Library Usage
------------
 
To be documented (please see ``ml/skylark_ml.cpp`` for a driver program).

Local Graph Computations
========================


Community Detection using Seed Nodes
------------------------------------

In community detection problems (i.e., graph clustering problems), one seeks to identify a set 
of nodes in a graph that are both internally cohesive and also well separated from the remainder 
of the graph. Such sets are then referred to as communities or clusters. In one important variant 
of community detection, the goal is to build a community around a given seed node or set of seed 
nodes. That is, the algorithm is given, as an input, a node (or nodes) in the graph, and the 
goal is to find a cluster in which it is a member.

The library implements the algorithm reported in the following paper:
 * | H. Avron and L. Horesh
   | Community Detection Using Time-Dependent Personalized PageRank

The interface is as follows:

.. cpp:function:: double FindLocalCluster(const GraphType& G, const std::unordered_set<typename GraphType::vertex_type>& seeds, std::unordered_set<typename GraphType::vertex_type>& cluster, double alpha, double gamma, double epsilon, int NX, bool recursive)

seeds is the set of input seeds, cluster is the set of output cluster. alpha, gamma, epsilon and NX are 
parameters of the algorithm. See paper for details. Defaults are specified.
If recursive is set to true (default is false)
the algorithm will recursively use the output cluster as seed until the cluster stops
improving (as measured using conductance).

The graph is specified using parameter G. The type is generic: the GraphType class is expected to
support the following:

.. cpp:type:: GraphType::vertex_type
Type of the graph nodes

.. cpp:function:: size_t GraphType::num_edges()
Return the number of edges in the graph.

.. cpp:function:: size_t GraphType::deg(vertex_type node)
Return the degree of the given node.

.. cpp:function:: iterator GraphType::adjanct_begin(vertex_type node)
Return an iterator to the start of the list of adjanct nodes of the input 
node. The iterator can be of any kind (must support increment, deref and comparison).

.. cpp:function:: iterator GraphType::adjanct_end(vertex_type node)
Return an iterator to the end of the list of adjanct nodes of the input 
node. 

See ``examples/community.cpp`` for an example of use.

Time-Dependent Personalized PageRank
------------------------------------

The community detection algorithm is based on a localized solution of 
a Time-Dependent Personlized PageRank diffusion problem. See the 
paper for details:

 * | H. Avron and L. Horesh
   | Community Detection Using Time-Dependent Personalized PageRank

The library also exposes the ability to solve the diffusion problem.
In this functionality, the input is a scalar function on nodes, and the 
output is a vector function on nodes. Each entry of the vector 
represents a different time point.

The interface is as follows:

.. cpp:function:: void TimeDependentPPR(const GraphType& G, const std::unordered_map<typename GraphType::vertex_type, T>& s, std::unordered_map<typename GraphType::vertex_type, El::Matrix<T> *>& y, El::Matrix<T> &x, double alpha, double gamma, double epsilon, int NX)

s is the input function of nodes, while y is the output. x specifies the time points which 
corresponds to the entries of x[node]. alpha, gamma, epsilon and NX are 
parameters of the algorithm. See paper for details. Defaults are specified.

The graph is specified using parameter G. The type is generic: the GraphType class is expected to
support the following:

.. cpp:type:: GraphType::vertex_type
Type of the graph nodes

.. cpp:function:: size_t GraphType::num_edges()
Return the number of edges in the graph.

.. cpp:function:: size_t GraphType::deg(vertex_type node)
Return the degree of the given node.

.. cpp:function:: iterator GraphType::adjanct_begin(vertex_type node)
Return an iterator to the start of the list of adjanct nodes of the input 
node. The iterator can be of any kind (must support increment, deref and comparison).

.. cpp:function:: iterator GraphType::adjanct_end(vertex_type node)
Return an iterator to the end of the list of adjanct nodes of the input 
node. 
