'''
IO facilities.

The following data formats are supported:
 * Hierarchical Data Format (HDF5)             : ``hdf5`` class
 * Matrix Market exchange format               : ``mtx`` class
 * Library for Supprort Vector Machines format : ``libsvm`` class
 * Raw text format                             : ``txt`` class

Three operations are exported, as appropriate:
 * ``read()``
 * ``write()``
 * ``stream()``

In all cases the constructor has as its first argument the filepath where the
data is (or is expected to be) for the operation to complete and as a last
argument a boolean flag for whether the operation will use parallel IO or
not.

``read()`` is typically provided the ``matrix_type`` of the object to load in memory
as its first argument. ``write()`` gets the matrix object itself as its only
argument.

Matrix types are identified with string identifiers that are self-explanatory:
 * ``'numpy-dense'``
 * ``'elemental-dense'``
 * ``'scipy-sparse'``
 * ``'combblas-sparse'``

.. note:: Parallel IO is experimental.

'''


# FIXME: IOError exceptions will be raised to indicate IO problems. Currently
# we expect the user to handle such cases which is typically the norm when
# issuing IO calls. Alternatively, we could catch them at this layer and throw
# them as skylark-specific exception objects.
class SkylarkIOTypeError(Exception):
    pass

import mpi4py.rc
mpi4py.rc.finalize = False
from mpi4py import MPI

import re
import scipy.sparse
import scipy.io
import numpy
import h5py
import elem


# TODO: Add support for parallel IO along the implementation show-cased in the
# wiki.
class hdf5(object):
    '''
    IO support for HDF5.

    * ``read()`` can load as ``'elemental-dense'`` and ``'numpy-dense'`` matrix types;
      for ``'elemental-dense'``, data ``distribution`` should also be indicated.
    * ``write()`` can save ``'elemental-dense'`` and ``'numpy-dense'`` matrix types.

    .. note::

     * Parallel IO requires a parallel-enabled build of the ``HDF5``
       library followed by compilation of ``h5py`` python bindings in MPI
       mode. These steps are detailed in
       http://www.h5py.org/docs/topics/mpi.html.

     * Parallel IO reads and writes matrix data in vectorized layout
       (column-major). The shape of the matrix is expected separately in
       ``'shape'`` dataset.
    '''
    def __init__(self, fpath, dataset='data', parallel=False, atomic=False):
        '''
        Class constructor.

        Parameters
        ----------
        fpath : string
         Filepath to read from or write to.

        dataset : string, optional
         HDF5 dataset string identifier; dataset is a collection of raw data
         elements and all associated metadata for operating on it.

        parallel: {False, True}
         Boolean flag whether the operation will use parallel IO or not.

        atomic: {False, True}
         Whether to use MPI atomic file access mode (only applicable for the
         case parallel=True.

        Returns
        -------
        hdf5 : object
         Ready to use object.
        '''
        self.fpath = fpath
        self.dataset = dataset
        self.parallel = parallel


    def _read_elemental_dense_parallel(self, distribution='MC_MR'):

        constructor = elemental_dense.get_constructor(distribution)

        f = h5py.File(self.fpath, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        height, width = int(f['shape'][0]), int(f['shape'][1])

        A = constructor()
        A.ResizeTo(height, width)
        local_height, local_width = A.LocalHeight, A.LocalWidth

        indices = elemental_dense.get_indices(A)
        local_data = f[self.dataset][indices]
        A.Matrix[:] = local_data.reshape(local_height, local_width, order='F')
        f.close()
        return A


    def _read_elemental_dense(self, distribution='MC_MR'):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        constructor = elemental_dense.get_constructor(distribution)
        # only the root process touches the filesystem
        if rank == 0:
            f = h5py.File(self.fpath, 'r')
            dataset_obj = f[self.dataset]
        shape = dataset_obj.shape if rank == 0 else None
        shape = comm.bcast(shape, root=0)
        height = shape[0]
        width = shape[1]

        num_entries = height * width
        # max memory capacity per process assumed/hardcoded to 10 blocks
        # XXX should this number be passed as a parameter?
        max_blocks_per_process = 10
        max_block_entries = int((1.0 * num_entries) / (max_blocks_per_process * size))

        # XXX We could set up a different block generating scheme, e.g. more
        # square-ish blocks
        block_height = int(numpy.sqrt(max_block_entries))
        while max_block_entries % block_height != 0:
            block_height = block_height + 1
        block_width = max_block_entries / block_height
        num_height_blocks = int(numpy.ceil(height / (1.0 * block_height)))
        num_width_blocks = int(numpy.ceil(width / (1.0 * block_width)))
        num_blocks = num_height_blocks * num_width_blocks

        A = constructor(height, width)
        for block in range(num_blocks):
            # the global coordinates of the block corners
            i_start = (block / num_width_blocks) * block_height
            j_start = (block % num_width_blocks) * block_width
            i_end = min(height, i_start + block_height)
            j_end = min(width, j_start + block_width)
            # the block size
            local_height = i_end - i_start
            local_width = j_end - j_start
            # [CIRC, CIRC] matrix is populated by the reader process (i.e. the root)...
            A_block = elem.DistMatrix_d_CIRC_CIRC(local_height, local_width)
            if rank == 0:
                A_block.Matrix[:] = dataset_obj[i_start:i_end, j_start:j_end]
            # ... then a view into the full matrix A is constructed...
            A_block_view = constructor()
            elem.View(A_block_view, A, i_start, j_start, local_height, local_width)
            # ... and finally this view is updated by redistribution of the [CIRC, CIRC] block
            elem.Copy(A_block, A_block_view)
        if rank == 0:
            f.close()
        return A


    def _read_numpy_dense(self):
        with h5py.File(self.fpath, 'r') as f:
            A = f[self.dataset].value
        return A


    def read(self, matrix_type='elemental-dense', distribution='MC_MR'):
        '''
        Read dataset from an HDF5 file as a matrix.

        Parameters
        ----------
        matrix_type : string, optional
         String identifier for the matrix object that is read in. Two options
         are available:

          * ``'elemental-dense'`` for DistMatrix<double,...> objects in Elemental
            library as wrapped in python (default).
          * ``'numpy-dense'`` for array objects in numpy package.

        distribution : string, optional
         String identifier for the matrix data distribution for the case the
         input matrix is required to be of ``'elemental-dense'`` type; this argument
         is ignored in the ``'numpy-dense'`` case. Available options are:

          * ``'MC_MR'`` (default)
          * ``'VC_STAR'``
          * ``'VR_STAR'``
          * ``'STAR_VC'``
          * ``'STAR_VR'``

        Returns
        -------
        matrix : Python-wrapped DistMatrix<double,...> or numpy array
         Matrix read.
        '''
        if matrix_type == 'elemental-dense':
            if self.parallel:
                A = self._read_elemental_dense_parallel(distribution)
            else:
                A = self._read_elemental_dense(distribution)
        elif matrix_type == 'numpy-dense':
            A = self._read_numpy_dense()
        return A


    def _write_numpy_dense(self, A):
        f = h5py.File(self.fpath, 'w')
        height, width = A.shape
        data = f.create_dataset(self.dataset, (height, width))
        data[:] = A
        f.close()


    def _write_elemental_dense_parallel(self, A):
        #distribution = elemental_dense.get_distribution(A)
        indices = elemental_dense.get_indices(A)

        height, width = A.Height, A.Width
        local_data = A.Matrix[:].ravel(order='F')
        f = h5py.File(self.fpath, 'w', driver='mpio', comm=MPI.COMM_WORLD)
        if self.atomic:
            f.atomic = True
        shape = f.create_dataset('shape', (2,), 'i')
        data = f.create_dataset(self.dataset, (height * width,))
        shape[0] = height
        shape[1] = width
        data[indices] = local_data
        f.close()


    def _write_elemental_dense(self, A):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # XXX currently gathers at root
        #FIXME: root not defined
        root = 0
        height, width = A.Height, A.Width
        A_CIRC_CIRC = elem.DistMatrix_d_CIRC_CIRC(height, width)
        elem.Copy(A, A_CIRC_CIRC)
        if root == 0:
            A_numpy_dense = A_CIRC_CIRC.Matrix[:]
            self._write_numpy_dense(A_numpy_dense)


    def write(self, A):
        '''
        Write matrix as a dataset to an HDF5 file.

        Parameters
        ----------
        A : Python-wrapped DistMatrix<double,...> or numpy array
         Matrix to write.

        Returns
        -------
        None
        '''
        if isinstance(A, numpy.ndarray):
            self._write_numpy_dense(A)
        else:
            if self.parallel:
                self._write_elemental_dense_parallel(A)
            else:
                self._write_elemental_dense(A)


class mtx(object):
    '''
    IO support for Matrix Market exchange format (MM).

    The ``layout`` of the data to is expected to be in ``'coordinate'`` or
    ``'array'`` MM formats.

     * ``read()`` can load ``'numpy-dense'``, ``'scipy-sparse'`` and
       ``'combblas-sparse'`` matrix types from ``'coordinate'`` MM format and
       ``'numpy-dense'``, ``'scipy-sparse'`` from ``'array'`` MM format.
     * ``write()`` can save in ``'array'`` and ``'coordinate'`` MM formats from
       ``'numpy-dense'`` and in ``'coordinate'`` MM format from
       ``'scipy-sparse'`` and ``'combblas-sparse'`` matrix types.
    '''
    def __init__(self, fpath, layout='coordinate', parallel=False):
        '''
        Class constructor.

        Parameters
        ----------
        fpath : string
         Filepath to read from or write to.

        layout : string, optional
         MM format, either ``'coordinate'`` (default) or ``'array'``

        parallel: {False, True}
         Boolean flag whether the operation will use parallel IO or not.

        Returns
        -------
        mtx : object
         Ready to use object.
        '''
        self.fpath = fpath
        self.layout = layout
        self.parallel = parallel


    def _read_numpy_dense(self):
        A = scipy.io.mmread(self.fpath)
        rows, cols, entries, fmt, field, symm = scipy.io.mminfo(self.fpath)
        if fmt == 'coordinate':
            A = A.toarray()
        return A


    def _read_scipy_sparse(self):
        A = scipy.io.mmread(self.fpath)
        rows, cols, entries, fmt, field, symm = scipy.io.mminfo(self.fpath)
        return scipy.sparse.csr_matrix(A)


    def _read_combblas_sparse(self):
        import kdt
        if self.parallel:
            A = kdt.Mat.load(self.fpath, par_IO=True)
        else:
            A = kdt.Mat.load(self.fpath)
        return A


    def read(self, matrix_type='scipy-sparse'):
        '''
        Read a matrix from an MM file.

        Parameters
        ----------
        matrix_type : string, optional
         String identifier for the matrix object that is read in. Three options
         are available:

          * ``'numpy-dense'`` for array objects in numpy package.
          * ``'scipy-sparse'`` for sparse array objects in scipy package,
            of type `scipy.sparse.csr_matrix` (default).
          * ``'combblas-sparse'`` for array objects in CombBLAS wrapped as python
            objects of `kdt.Mat` type.

        Returns
        -------
        matrix : numpy array or scipy sparse array or CombBLAS kdt.Mat matrix
         Matrix read.
        '''
        if matrix_type == 'numpy-dense':
            A = self._read_numpy_dense()
        if matrix_type == 'scipy-sparse':
            A = self._read_scipy_sparse()
        elif matrix_type == 'combblas-sparse':
            A = self._read_combblas_sparse()
        return A


    def _write_numpy_dense_coordinate(self, A):
        values = A.flatten()
        n = len(values)
        rows = range(n)
        cols = [0 for i in range(n)]
        vector = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(n, 1))
        scipy.io.mmwrite(self.fpath, vector)


    def _write_numpy_dense_array(self, A):
        scipy.io.mmwrite(self.fpath, A)


    def _write_numpy_dense(self, A):
        if self.layout == 'coordinate':
            self._write_numpy_dense_coordinate(A)
        elif self.layout == 'array':
            self._write_numpy_dense_array(A)


    def _write_scipy_sparse(self, A):
        if self.layout == 'coordinate':
            scipy.io.mmwrite(self.fpath, A)


    def _write_combblas_sparse(self, A):
        if self.layout == 'coordinate':
            A.save(self.fpath)


    def write(self, A):
        '''
        Write matrix to an MM file.

        Parameters
        ----------
        matrix : numpy array or scipy sparse array or CombBLAS kdt.Mat matrix
         Matrix to write.

        Returns
        -------
        None
        '''
        if isinstance(A, numpy.ndarray):
            self._write_numpy_dense(A)
        elif isinstance(A, scipy.sparse.csr_matrix):
            self._write_scipy_sparse(A)
        else:
            import kdt
            if isinstance(A, kdt.Mat):
                self._write_combblas_sparse(A)
            else:
                raise SkylarkIOTypeError("Cannot handle write with matrix type " + str(type(A)))


class libsvm(object):
    '''
    IO support for libsvm.

    A libsvm file for our purposes consists of lines. Each line corresponds to
    an instance: this starts with a *label* (the class label of the instance)
    and it is followed by the set of its *feature*/*value* pairs.

    ``read()`` and ``stream()`` operations are supported that produce
    (sparse-matrix, vector) pairs. The sparse matrix uses the *features* as
    column indices, the *values* as matrix entries and a running *index* of
    the instances as row indices. The vector accumulates the *labels*.

    So there are four vectors expected from the parsing stage:
     * *labels*
     * *indices*
     * *features*
     * *values*

    and these are returned from ``read_vectors()`` and ``stream_vectors()``
    methods (``vectors_to_matrices()`` provides the conversion).

    In streaming, the file is not read in one shot (as it happens e.g. in
    ``read()``), but in stages: each stage consumes the next ``block_size``
    instances (i.e lines) in the input file (or the number of lines left,
    whichever happens to be smallest), thus dramatically bounding overall memory
    requirements.
    '''

    def __init__(self, fpath, parallel=False):
        '''
        Class constructor.

        Parameters
        ----------
        fpath : string
         Filepath to read from or write to.

        parallel: {False, True}
         Boolean flag whether the operation will use parallel IO or not.

        Returns
        -------
        libsvm : object
         Ready to use object.
        '''
        self.fpath = fpath
        self.parallel = parallel


    def read_vectors(self):
        '''
        Read the labels, indices, features and values from a libsvm file as a
        tuple of vectors.

        Parameters
        ----------
        None

        Returns
        -------
        labels    : list of integers
         Class labels of instances.

        indices   : list of integers
         Row indices of features/values in their sparse matrix representation.

        features  : list of integers
         Column indices of fetures/values in their sparse matrix
        representation.

        values    : list of floats
         Sparse matrix entries; values for features.
        '''
        f = open(self.fpath, 'r')
        labels = []
        indices = []
        features = []
        values = []
        for (index, line) in enumerate(f):
            _l, _i, _f, _v = self._parse(index, line)
            labels.append(_l)
            indices.extend(_i)
            features.extend(_f)
            values.extend(_v)
        f.close()
        return labels, indices, features, values


    def stream_vectors(self, block_size=1000):
        '''
        Stream the labels, indices, features and values from a libsvm file as a
        tuple of vectors.

        Parameters
        ----------
        block_size : int, optional
         Number of lines to parse in each iteration.

        Returns
        -------
        generator
         A function that behaves like an iterator over the tuples of vectors for
        the libsvm parts (of ``block_size`` lines max each).
        '''

        f = open(self.fpath, 'r')
        has_lines = True
        while has_lines:
            labels = []
            indices = []
            features = []
            values = []
            index = 0
            block_complete = False
            for line in f:
                _l, _i, _f, _v = self._parse(index, line)
                labels.append(_l)
                indices.extend(_i)
                features.extend(_f)
                values.extend(_v)
                index += 1
                if index == block_size:
                    block_complete = True
                    break
            if not block_complete:
                has_lines = False
            yield labels, indices, features, values
        f.close()


    def read(self):
        '''
        Read the (features matrix, labels vector) pair from a libsvm file.

        Parameters
        ----------
        None

        Returns
        -------
        matrix : scipy sparse array
         Features matrix read.

        vector : numpy array
         Labels vector read.
        '''

        vectors = self.read_vectors()
        matrices = self.vectors_to_matrices(vectors)
        return matrices


    def stream(self, num_features, block_size=1000):
        '''
        Stream the (features matrix, labels vector) pair from a libsvm file.

        Parameters
        ----------
        num_features : int
         Number of features.

        block_size : int, optional
         Number of lines to parse in each iteration.


        Returns
        -------
        generator
         A function that behaves like an iterator over the libsvm parts (of
         `block_size` lines max each).

        '''

        for vectors in self.stream_vectors(block_size):
            yield self.vectors_to_matrices(vectors, num_features)


    def vectors_to_matrices(self, vectors, num_features=0):
        '''
        Parameters
        ----------
        vectors : tuple of lists
         (labels, indices, features, values); see ``read_vector()`` for
         decriptions.

        num_features : int, optional
         Number of features.

        Returns
        -------
        generator
         A function that behaves like an iterator over the libsvm parts (of
         ``block_size`` lines max each).
        '''

        labels, indices, features, values = vectors
        num_samples = indices[-1] + 1
        num_features = max(max(features) + 1, num_features)
        feature_matrix = scipy.sparse.csr_matrix(
            (values, (indices, features)),
            shape=(num_samples, num_features))
        label_matrix = numpy.asarray(labels)
        return feature_matrix, label_matrix


    def _parse(self, index, line):
        tokens = re.split(re.compile('[:\s]'), line.strip())
        label = float(tokens[0])
        features = map(int, tokens[1::2])
        values = map(float, tokens[2::2])
        indices = [index] * len(features)
        return label, indices, features, values


class txt(object):
    '''
    IO support for raw text format.

    * ``read()`` can load as ``'numpy-dense'`` matrix type.
    * ``write()`` can save ``'elemental-dense'`` and ``'numpy-dense'`` matrix types.
    '''

    def __init__(self, fpath, parallel=False):
        '''
        Class constructor.

        Parameters
        ----------
        fpath : string
         Filepath to read from or write to.

        parallel: {False, True}
         Boolean flag whether the operation will use parallel IO or not.

        Returns
        -------
        txt : object
         Ready to use object.
        '''
        self.fpath = fpath
        self.parallel = parallel

    def _read_numpy_dense(self):
        A = numpy.loadtxt(self.fpath)
        return A

    def read(self, matrix_type='numpy-dense'):
        '''
        Read dataset from an raw text file as a matrix.

        Parameters
        ----------
        matrix_type : string, optional
         String identifier for the matrix object that is read in. One option
          * ``'numpy-dense'`` for array objects in numpy package (default).
          * 'asasas'
        hello : string
         hi
        '''
        if matrix_type == 'numpy-dense':
            A = self._read_numpy_dense()
        else:
            raise SkylarkIOTypeError("Cannot reader matrix of type " + matrix_type)
        return A

    def _write_numpy_dense(self, A):
        numpy.savetxt(self.fpath, A)

    def _write_elemental_dense(self, A):
        elem.Write(A, '', self.fpath)

    def write(self, A):
        '''
        Write matrix in raw text format.

        Parameters
        ----------
        A : Python-wrapped DistMatrix<double,...> or numpy array
         Matrix to write.

        Returns
        -------
        None
        '''

        if isinstance(A, numpy.ndarray):
            self._write_numpy_dense(A)
        else:
            self._write_elemental_dense(A)



class elemental_dense(object):
    '''
    Utility functions for ``'elemental-dense'`` matrices.
    '''

    _constructors = {
        'MC_MR'   : elem.DistMatrix_d,
        'VC_STAR' : elem.DistMatrix_d_VC_STAR,
        'VR_STAR' : elem.DistMatrix_d_VR_STAR,
        'STAR_VC' : elem.DistMatrix_d_STAR_VC,
        'STAR_VR' : elem.DistMatrix_d_STAR_VR
        }

    @classmethod
    def get_distribution(cls, A):
        '''
        String identifier of matrix distribution.

        Parameters
        ----------
        A : ``'elemental-dense'`` matrix
         Input matrix.

        Returns
        -------
        distribution : string
         String identifier of matrix distribution.
        '''
        for (key, value) in cls._constructors.iteritems():
            if type(A) == value:
                distribution = key
        return distribution


    @classmethod
    def get_constructor(cls, distribution='MC_MR'):
        '''
        Constructor for a matrix distribution.

        Parameters
        ----------
        distribution : string
         String identifier of matrix distribution.

        Returns
        -------
        constructor : object
         Constructor; ``constructor()`` will instantiate the matrix.
        '''

        return cls._constructors[distribution]


    @classmethod
    def get_indices(cls, A):
        '''
        Matrix indices.

        Parameters
        ----------
        A : ``'elemental-dense'`` matrix
         Input matrix.

        Returns
        -------
        indices : list of integers
         Global indices into the the vector of entries of matrix A that are
         locally hosted if the matrix is traversed in column-major mode.

        '''

        distribution = cls.get_distribution(A)
        if distribution == 'MC_MR':
            indices = cls._indices_MC_MR(A)
        elif distribution == 'VC_STAR':
            indices = cls._indices_VC_STAR(A)
        elif distribution == 'VR_STAR':
            indices = cls._indices_VR_STAR(A)
        elif distribution == 'STAR_VC':
            indices = cls._indices_STAR_VC(A)
        elif distribution == 'STAR_VR':
            indices = cls._indices_STAR_VR(A)
        return indices


    @staticmethod
    def _indices_MC_MR(A):
        vc_rank = A.Grid.VCRank
        col_alignment = A.ColAlignment
        row_alignment = A.RowAlignment
        col_stride = A.ColStride
        row_stride = A.RowStride
        col_shift = A.ColShift
        row_shift = A.RowShift
        height = A.Height
        width = A.Width
        local_height = A.LocalHeight
        local_width = A.LocalWidth
        local_size = local_height * local_width

        indices = [0 for i in range(local_size)]
        for j in range(width):
            for i in range(height):
                owner_row = (i + col_alignment) % col_stride
                owner_col = (j + row_alignment) % row_stride
                owner_rank = owner_row + owner_col * col_stride
                if vc_rank == owner_rank:
                    i_loc = (i - col_shift) / col_stride
                    j_loc = (j - row_shift) / row_stride
                    global_index = j * height + i
                    local_index = j_loc * local_height + i_loc
                    indices[local_index] = global_index
        return indices


    @staticmethod
    def _indices_VC_STAR(A):
        vc_rank = A.Grid.VCRank
        grid_size = A.Grid.Size
        col_alignment = A.ColAlignment
        col_shift = A.ColShift
        height = A.Height
        width = A.Width
        local_height = A.LocalHeight
        local_width = A.LocalWidth
        local_size = local_height * local_width

        indices = [0 for i in range(local_size)]
        for i in range(height):
            owner_rank = (i + col_alignment) % grid_size
            if vc_rank == owner_rank:
                i_loc = (i - col_shift) / grid_size
                for j in range(width):
                    j_loc = j
                    global_index = j * height + i
                    local_index = j_loc * local_height + i_loc
                    indices[local_index] = global_index
        return indices


    @staticmethod
    def _indices_VR_STAR(A):
        vr_rank = A.Grid.VRRank
        grid_size = A.Grid.Size
        col_alignment = A.ColAlignment
        col_shift = A.ColShift
        height = A.Height
        width = A.Width
        local_height = A.LocalHeight
        local_width = A.LocalWidth
        local_size = local_height * local_width

        indices = [0 for i in range(local_size)]
        for i in range(height):
            owner_rank = (i + col_alignment) % grid_size
            if vr_rank == owner_rank:
                i_loc = (i - col_shift) / grid_size
                for j in range(width):
                    j_loc = j
                    global_index = j * height + i
                    local_index = j_loc * local_height + i_loc
                    indices[local_index] = global_index
        return indices


    @staticmethod
    def _indices_STAR_VC(A):
        vc_rank = A.Grid.VCRank
        grid_size = A.Grid.Size
        row_alignment = A.RowAlignment
        row_shift = A.RowShift
        height = A.Height
        width = A.Width
        local_height = A.LocalHeight
        local_width = A.LocalWidth
        local_size = local_height * local_width

        indices = [0 for i in range(local_size)]
        for j in range(width):
                owner_rank = (j + row_alignment) % grid_size
                if vc_rank == owner_rank:
                    j_loc = (j - row_shift) / grid_size
                    for i in range(height):
                        i_loc = i
                        global_index = j * height + i
                        local_index = j_loc * local_height + i_loc
                        indices[local_index] = global_index
        return indices


    @staticmethod
    def _indices_STAR_VR(A):
        vr_rank = A.Grid.VRRank
        grid_size = A.Grid.Size
        row_alignment = A.RowAlignment
        row_shift = A.RowShift
        height = A.Height
        width = A.Width
        local_height = A.LocalHeight
        local_width = A.LocalWidth
        local_size = local_height * local_width

        indices = [0 for i in range(local_size)]
        for j in range(width):
                owner_rank = (j + row_alignment) % grid_size
                if vr_rank == owner_rank:
                    j_loc = (j - row_shift) / grid_size
                    for i in range(height):
                        i_loc = i
                        global_index = j * height + i
                        local_index = j_loc * local_height + i_loc
                        indices[local_index] = global_index
        return indices



def _usage_tests(usps_path='./datasets/usps.t'):
    '''
    Various simple example scenaria for showing the usage of the IO facilities
    '''

    ############################################################
    # libsvm
    ############################################################
    fpath = usps_path

    # read features matrix and labels vector
    try:
        store = libsvm(fpath)
    except ImportError:
        print 'Please provide the path to usps.t as an argument'
        import sys; sys.exit()
    features_matrix, labels_matrix = store.read()
    matrix_info = features_matrix.shape, features_matrix.nnz, labels_matrix.shape

    # stream features matrix and labels vector
    store = libsvm(fpath)
    for features_matrix, labels_matrix in store.stream(num_features=400, block_size=100):
        matrix_info = features_matrix.shape, features_matrix.nnz, labels_matrix.shape
    print 'libsvm OK'

    ############################################################
    # mtx
    ############################################################
    features_fpath = '/tmp/test_features.mtx'
    labels_fpath   = '/tmp/test_labels.mtx'

    # write features and labels
    store = mtx(features_fpath)
    store.write(features_matrix)
    store = mtx(labels_fpath)
    store.write(labels_matrix)

    # read back features as 'scipy-sparse' and 'combblas-sparse'
    store = mtx(features_fpath)
    A = store.read('scipy-sparse')
    store = mtx(features_fpath)
    B = store.read('combblas-sparse')
    print 'mtx OK'

    ############################################################
    # hdf5
    ############################################################
    fpath = '/tmp/test_matrix.h5'

    # write a random 'numpy-dense' to HDF5 file
    store = hdf5(fpath)
    A = numpy.random.random((20, 65))
    store.write(A)

    # read HDF5 file as:
    # - 'numpy-dense'
    # - 'elemental-dense' (default 'MC_MR' distribution)
    # - 'elemental-dense' ('VC_STAR' distribution)
    B = store.read('numpy-dense')
    C = store.read('elemental-dense')
    D = store.read('elemental-dense', distribution='VC_STAR')
    print 'hdf OK'

    ############################################################
    # txt
    ############################################################
    fpath = '/tmp/test_matrix.txt'

    # write a uniform random 'elemental-dense', 'MC_MR' distribution
    A = elem.DistMatrix_d()
    elem.Uniform(A, 10, 30)
    store = txt(fpath)
    store.write(A)

    # read the matrix back as 'numpy-dense'
    store = txt(fpath)
    A = store.read('numpy-dense')
    print 'txt OK'

if __name__ == '__main__':
    _usage_tests()
