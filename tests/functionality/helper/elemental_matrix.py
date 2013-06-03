import elem

def create_elemental_matrix(height = 10, width = 5):
    """
    Create a matrix to sketch
    """

    grid = elem.Grid()
    A    = elem.DistMat_VR_STAR( grid )
    A.Resize(height, width)

    localHeight = A.LocalHeight()
    localWidth  = A.LocalWidth()
    colShift    = A.ColShift()
    rowShift    = A.RowShift()
    colStride   = A.ColStride()
    rowStride   = A.RowStride()
    data        = A.Data()
    ldim        = A.LDim()

    for jLocal in xrange(0, localWidth):
        j = rowShift + jLocal * rowStride
        for iLocal in xrange(0, localHeight):
            i = colShift + iLocal * colStride
            data[iLocal + jLocal * ldim] = (i - j) / (localWidth * localHeight)

    return A
