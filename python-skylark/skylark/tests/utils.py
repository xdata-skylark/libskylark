import El
import numpy as np
import skylark.io as sl_io


# Util functions useful to test the library

def load_libsvm_file(fpath, col_row):
    """
    Read from a libsv file
    
    :param fpath: path of the file
    :param col_row: 0 to read in col mode, 1 to read in row mode
    :returns: Returns X, Y matrices
    """
    # Read the data
    X = El.DistMatrix()
    Y = El.DistMatrix()
    return sl_io.readlibsvm(fpath, X, Y, 0)

def to_num_py(M):
    """
    Returns the numpy representation of M 
    
    :param M: Matrix (numpy, or Elemental)
    :returns: Returns the numpy representation of M
    """
    if isinstance(M, El.Matrix):
        return M.ToNumPy()
    elif (isinstance(M, El.DistMatrix)):
        return M.Matrix().ToNumPy()
    return M

def equal(A, B, threshold=1.e-4):
    """
    Returns if A and B are equal (given a threshold).
    
    :param A: Matrix
    :param B: Matrix 
    :param threshold: max err between elements default value 1.e-4 
    :returns: if M looks like a kerner or not
    """
    A = to_num_py(A)
    B = to_num_py(B)
    return ((A - B) < threshold).all()

def is_kernel(M, positive=True, threshold=1.e-4):
    """
    Cheking if matrix M COULD be a kernel.
    Kernels look like:
        - All the values are >= 0 (optional?)
        - M[i,j] == M[j, i] (They are symmetric)
        - M[i,j]^2 <= M[i,i]*M[j,j] Cauchy-Schwarz Inequality (TODO)

    :param M: Matrix
    :param positive: should check all the entries are positive?
    :param threshold: when are two numbers equal (diff)
    :returns: if M looks like a kerner or not
    """

    M = to_num_py(M)
 
    # Symmetric and C-S Inequality
    for i, j in np.ndindex(M.shape):
        # Are all the entries positive?
        if positive and (M[i][j] < 0 and not abs(M[i][j]) < threshold):
            return False
        # Is Symmetric?
        if  abs(M[i,j] - M[j,i]) > threshold:
            return False
        # TODO: Check Cauchy-Schwarz Inequality
        #we don't have acces to u and v to compute ||u||*||v||!  
        # elif not (CS Inequality):
        #   return false
        
    return True
