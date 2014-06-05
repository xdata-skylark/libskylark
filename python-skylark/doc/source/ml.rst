Machine Learning
*****************



Randomized Kernel Methods
==========================



Commandline Usage
==================

::

    Training mode usage: skylark_ml [options] --trainfile trainfile --modelfile modelfile
    Testing mode usage: skylark_ml --modelfile modelfile --testfile testfile
      -h [ --help ]                         produce a help message
      -l [ --lossfunction ] arg (=0)        Loss function (0:SQUARED (L2), 1:LAD 
					    (L1), 2:HINGE, 3:LOGISTIC)
      -r [ --regularizer ] arg (=0)         Regularizer (0:L2, 1:L1)
      -k [ --kernel ] arg (=0)              Kernel (0:LINEAR, 1:GAUSSIAN, 
					    2:POLYNOMIAL, 3:LAPLACIAN, 
					    4:EXPSEMIGROUP)
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
      --regular arg (=1)                    Default is to use 'fast' feature 
					    mapping, if available.Use this flag to 
					    force regular mapping (default: false)
      --cachetransforms arg (=0)            Default is to not cache feature 
					    transforms per iteration, but generate 
					    on fly. Use this flag to force 
					    transform caching if you have enough 
					    memory (default: false)
      --fileformat arg (=0)                 Fileformat (default: 0 (libsvm->dense),
					    1 (libsvm->sparse), 2 (hdf5->dense), 3 
					    (hdf5->sparse)
      -i [ --MAXITER ] arg (=100)           Maximum Number of Iterations (default: 
					    100)
      --trainfile arg                       Training data file (required in 
					    training mode)
      --modelfile arg                       Model output file
      --valfile arg                         Validation file (optional)
      --testfile arg                        Test file (optional in training mode; 
					    required in testing mode)


