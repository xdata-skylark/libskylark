import elem
import numpy

def local2distributed(A, type=elem.DistMatrix_d):
    """
    Takes an NumPy local matrix and returns an Elemental distribute matrix.
    The type of the Elemental is given by type, whose default is DistMatrix_d.
    Important:
    - The NumPy matrix should be present on all ranks.
    - Elemental default matrix generation is used.
    """
    Adist = type(A.shape[0], A.shape[1])
    localHeight = Adist.LocalHeight
    localWidth = Adist.LocalWidth
    colShift = Adist.ColShift
    rowShift = Adist.RowShift
    colStride = Adist.ColStride
    rowStride = Adist.RowStride
    data = Adist.Matrix
    ldim = Adist.LDim
    for jLocal in xrange(0,localWidth):
        j = rowShift + jLocal*rowStride
        for iLocal in xrange(0,localHeight):
            i = colShift + iLocal*colStride
            data[iLocal, jLocal] = A[iLocal, jLocal]
    
    return Adist
