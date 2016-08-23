import El
import numpy

def create_elemental_matrix(m, n, f, type=El.DistMatrix):
    """
    Create an Elemental matrix by setting position (i, j) to f(i, j).
    """
    Adist = type()
    Adist.Resize(m, n)
    localHeight = Adist.LocalHeight()
    localWidth = Adist.LocalWidth()
    colShift = Adist.ColShift()
    rowShift = Adist.RowShift()
    colStride = Adist.ColStride()
    rowStride = Adist.RowStride()
    data = Adist.Matrix()
    ldim = Adist.LDim()
    for jLocal in xrange(0,localWidth):
        j = rowShift + jLocal*rowStride
        for iLocal in xrange(0,localHeight):
            i = colShift + iLocal*colStride
            Adist.SetLocal(iLocal, jLocal, f(i, j))

    return Adist

def local2distributed(A, type=El.DistMatrix):
    """
    Takes an NumPy local matrix and returns an Elemental distribute matrix.
    The type of the Elemental is given by type, whose default is DistMatrix_d.
    Important:
    - The NumPy matrix should be present on all ranks.
    - Elemental default matrix generation is used.
    """
    Adist = create_elemental_matrix(A.shape[0], A.shape[1],\
                                        lambda i,j : A[i, j], \
                                        type);

    return Adist

