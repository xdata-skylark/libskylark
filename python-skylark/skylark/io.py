"""
Created on April 8, 2013.
@author: Vikas Sindhwani (vsindhw@us.ibm.com)
 
    Example command line Usage for converting datasets into HDF5 format.

    >>> python skylark/io.py --libsvm2hdf5 datasets/usps.t datasets/usps.hdf5
    
"""
import h5py
import elem
import numpy, scipy, scipy.sparse
import argparse

class hdf5(object):
    """
    Class for reading a Dense matrix from a dataset in an HDF5 file in parallel into an Elemental VC,* matrix,
    or for writing a Numpy array into a dataset in an HDF5 file. Note that HDF5 reads in row-major (C-style) orientation
    while Elementals local buffers are column-major. For this reason, we assume that the HDF5 file has a transposed version of the dataset.
    Therefore, Elementals view of the data is the transposed version of what you will see with tools like h5dump.  
        
    """
    def __init__(self, filename, mode='r'):
        """
        reader = hdf5(filename, mode='r', dataset = 'data')
        
        Creates an hdf5 IO object.
        
        Parameters
        -----------
        filename : path to an HDF5 file
        mode : 'r' | 'w'
        inputdataset : Name of the dataset specifying an n x d input feature matrix.
        outputdataset : Name of the dataset specifying n x k label matrix.

        Returns
        ---------
        A reader object.
        
        """
        self.filename = filename
        self.file = h5py.File(filename, mode)       
            
    def close(self):
        """
        close(self)
        
        Close the file.
        """
        self.file.close()
        
    def dimensions(self, dataset = 'data'):
        """
        dimensions(self, dataset = "data")
        
        Return dimensions of a dataset.
        
        Parameters
        -----------
        dataset : name of the dataset 
        
        Returns
        ---------
        tuple (m,n) for an m x n matrix
        """
        data = self.file[dataset]
        (n, m) = data.shape
        return (m, n)
    
    def write_dense(self, A, dataset = 'data'):
        """
        write_dense(self, A, dataset = "data")
        
        Writes a Numpy array to the specified dataset in the HDF5 file 
        
        Parameters
        -----------
        A : A numpy array
        dataset : name of the dataset
        
        Returns
        --------
        Nothing.
        """
        dset = self.file.create_dataset(dataset, data=numpy.transpose(A))
        return
    
    
    def read_dense(self, A, dataset= 'data'):
        """
        read_dense(self, A, dataset= "data")
         
        Populates the local chunk of an Elemental distributed [VC, *] matrix A
        
        Parameters
        --------------
        A : Elemental VC,* matrix.
        dataset: name of the dataset.
        
        Returns
        ----------
        Nothing.
        """
        # Get a dataset object associated with the dataset name
        dataset = self.file[dataset]
        
        # Get the Elemental local matrix buffer
        buffer = A.Data()
        
        if A.LocalHeight()>0:
            space_id = dataset.id.get_space()
            space_id.select_hyperslab((0, A.ColShift()), (A.LocalWidth(), A.LocalHeight()), stride=(1, A.ColStride()))
            m = A.LocalHeight()
            n = A.LocalWidth()
            space_id2 = h5py.h5s.create_simple((m*n,1))
            dataset.id.read(space_id2, space_id, buffer)
        
        return

    def close(self):
        self.file.close()
        
def sparselibsvm(filename):
        """
        sparselibsvm(filename)
     
        Reads data from a libsvm file into scipy.sparse.csr_matrix data structure.
     
        Parameters
        -------------
        filename
    
        Returns
        ----------
        tuple (X,Y) where X is sparse.csr_matrix and Y is a list of labels.
        """
        Y = []
        I = []
        J = []
        V = []
        rowid = 0
        for line in open(filename):
                tokens = line.strip().split()
                for tok in tokens:
                        if tok.find(':')>0:
                                ind, val = tok.split(':')
                                V.append(float(val))
                                J.append(int(ind)-1)
                                I.append(rowid)
                            
                        else:
                                Y.append(float(tok))
                rowid = rowid  + 1
        nfeatures = max(J) + 1        
        X = scipy.sparse.csr_matrix( (V, (I, J)), shape = (rowid,nfeatures))
        return (X,Y)


def libsvm2hdf5(filename, outfilename):
    """
    libsvm2hdf5(filename, outfilename)
    
    Reads data from filename in libsvm format and dumps an hdf5 file outfilename
    with datasets "Features" and "Labels"
    
    """
    print "Reading %s" % filename
    (X,Y) = sparselibsvm(filename)
    writer = hdf5(outfilename, "w")
    print "Writing dataset: Features and dataset: Labels into %s" % outfilename
    writer.write_dense(X.todense(), 'Features')
    writer.write_dense(Y, 'Labels')
    writer.close()
    

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Skylark IO Utility')
    parser.add_argument("--libsvm2hdf5", action='store_true')
    parser.add_argument("libsvmfile")
    parser.add_argument("outputHDF5file")
      
    args = parser.parse_args()
    libsvm2hdf5(args.libsvmfile, args.outputHDF5file)
        