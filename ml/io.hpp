/*
 * io.hpp
 *
 *  Created on: Feb 7, 2014
 *      Author: vikas
 */

#ifndef IO_HPP_
#define IO_HPP_

#include <boost/mpi.hpp>
#include <sstream>
#include <cstdlib>
#include <string>
#include <elemental.hpp>
#include "../base/sparse_matrix.hpp"

using namespace std;
namespace bmpi =  boost::mpi;

typedef skylark::sketch::context_t skylark_context_t;
typedef elem::DistMatrix<double, elem::CIRC, elem::CIRC> DistCircMatrixType;
typedef skylark::base::sparse_matrix_t<double> sparse_matrix_t;


#ifdef SKYLARK_HAVE_HDF5
#include <H5Cpp.h>

int write_elem_hdf5(string fName, elem::Matrix<double>& X,
        elem::Matrix<double>& Y) {

    try {

        cout << "Writing to file " << fName << endl;

        H5::Exception::dontPrint();

        H5::H5File file( fName, H5F_ACC_TRUNC );
        hsize_t dimsf[2]; // dataset dimensions
        dimsf[0] = X.Width();
        dimsf[1] = X.Height();
        H5::DataSpace dataspace( 2, dimsf );

       H5::FloatType datatype( H5::PredType::NATIVE_DOUBLE );
       datatype.setOrder( H5T_ORDER_LE );
       /*
       * Create a new dataset within the file using defined dataspace and
       * datatype and default dataset creation properties.
       */
       H5::DataSet dataset = file.createDataSet( "X", datatype, dataspace );
       /*
       * Write the data to the dataset using default memory space, file
       * space, and transfer properties.
       */
       cout << "Writing X" << endl;
       dataset.write( X.Buffer(), H5::PredType::NATIVE_DOUBLE );


       hsize_t dimsf2[1]; // dataset dimensions
       dimsf2[0] = Y.Height();
       H5::DataSpace dataspace2( 1, dimsf2 );


       H5::DataSet dataset2 = file.createDataSet( "Y", datatype, dataspace2 );
              /*
              * Write the data to the dataset using default memory space, file
              * space, and transfer properties.
              */
       cout << "Writing Y" << endl;
       dataset2.write( Y.Buffer(), H5::PredType::NATIVE_DOUBLE );


    }
   // catch failure caused by the H5File operations
      catch( H5::FileIException error )
      {
      error.printError();
      return -1;
      }
      // catch failure caused by the DataSet operations
      catch( H5::DataSetIException error )
      {
      error.printError();
      return -1;
      }
      // catch failure caused by the DataSpace operations
      catch( H5::DataSpaceIException error )
      {
      error.printError();
      return -1;
      }
      // catch failure caused by the DataSpace operations
      catch( H5::DataTypeIException error )
      {
      error.printError();
      return -1;
      }
  return 0; // successfully terminated
}

void read_hdf5_dense(skylark_context_t& context, string fName,
        sparse_matrix_t& X,
        elem::Matrix<double>& Y, int blocksize = 10000) {

    // Not Implemented.
}

void read_hdf5_dense(skylark_context_t& context, string fName,
        elem::DistMatrix<double, elem::STAR, elem::VC>& X,
        elem::DistMatrix<double, elem::VC, elem::STAR>& Y, int blocksize = 10000) {

        bmpi::timer timer;
        if (context.rank==0)
                    cout << "Reading from file " << fName << endl;


       H5::H5File file( fName, H5F_ACC_RDONLY );
       H5::DataSet datasetX = file.openDataSet( "X" );
       H5::DataSpace filespaceX = datasetX.getSpace();
       int rank = filespaceX.getSimpleExtentNdims();
       hsize_t dimsX[2]; // dataset dimensions
       rank = filespaceX.getSimpleExtentDims( dimsX );
       hsize_t n = dimsX[0];
       hsize_t d = dimsX[1];

       H5::DataSet datasetY = file.openDataSet( "Y" );
       H5::DataSpace filespaceY = datasetY.getSpace();
       hsize_t dimsY[1]; // dataset dimensions
       rank = filespaceX.getSimpleExtentDims( dimsY );

       hsize_t countX[2];
       hsize_t countY[1];

        int numblocks = ((int) n/ (int) blocksize); // of size blocksize
        int leftover = n % blocksize;
        int block = blocksize;

        hsize_t offsetX[2], offsetY[1];

//        if (min_d > 0)
 //                   d = std::max(d, min_d);

        X.Resize(d, n);
        Y.Resize(n,1);


        for(int i=0; i<numblocks+1; i++) {

            if (i==numblocks)
                block = leftover;
            if (block==0)
                break;

            DistCircMatrixType x(d, block), y(block, 1);
            x.SetRoot(0);
            y.SetRoot(0);
            elem::MakeZeros(x);


            offsetX[0] = i*blocksize;
            offsetX[1] = 0;
            countX[0] = block;
            countX[1] = d;
            offsetY[0] = i*blocksize;
            countY[0] = block;


            filespaceX.selectHyperslab( H5S_SELECT_SET, countX, offsetX );
            filespaceY.selectHyperslab( H5S_SELECT_SET, countY, offsetY );

            if(context.rank==0) {
                cout << "Reading and distributing chunk " << i*blocksize << " to " << i*blocksize + block - 1 << " ("<< block << " elements )" << endl;

                double *Xdata = x.Matrix().Buffer();
                double *Ydata = y.Matrix().Buffer();

                dimsX[0] = block;
                dimsX[1] = d;
                H5::DataSpace mspace1(2, dimsX);
                datasetX.read( Xdata, H5::PredType::NATIVE_DOUBLE, mspace1, filespaceX );

                dimsY[0] = block;
                H5::DataSpace mspace2(1,dimsY);
                datasetY.read( Ydata, H5::PredType::NATIVE_DOUBLE, mspace2, filespaceY );

            }

            elem::DistMatrix<double, elem::STAR, elem::VC> viewX;
            elem::DistMatrix<double, elem::VC, elem::STAR> viewY;

            elem::View(viewX, X, 0, i*blocksize, x.Height(), x.Width());
            elem::View(viewY, Y, i*blocksize, 0, x.Width(), 1);

            viewX = x;
            viewY = y;

        }

        double readtime = timer.elapsed();
        if (context.rank==0)
                cout << "Read Matrix with dimensions: " << n << " by " << d << " (" << readtime << "secs)" << endl;

}
#endif

void read_libsvm(skylark_context_t& context, string fName,
		elem::DistMatrix<double, elem::STAR, elem::VC>& X,
		elem::DistMatrix<double, elem::VC, elem::STAR>& Y,
		int min_d = 0, int blocksize = 10000) {
	if (context.rank==0)
			cout << "Reading from file " << fName << endl;

	ifstream file(fName.c_str());
	string line;
	string token, val, ind;
	float label;
	unsigned int start = 0;
	unsigned int delim, t;
	int n = 0;
	int d = 0;
	int i, j, last;
	char c;

	bmpi::timer timer;


	// make one pass over the data to figure out dimensions - will pay in terms of preallocated storage.
	if (context.rank==0) {
	    while(!file.eof()) {
	        getline(file, line);
	        if(line.length()==0)
	            break;
	        delim = line.find_last_of(":");
	        if(delim > line.length())
	            continue;
	        n++;
	        t = delim;
	        while(line[t]!=' ') {
	            t--;
	        }
	        val = line.substr(t+1, delim - t);
	        last = atoi(val.c_str());
	        if (last>d)
	            d = last;
	    }
	    if (min_d > 0)
	        d = std::max(d, min_d);

	    // prepare for second pass
	    file.clear();
	    file.seekg(0, std::ios::beg);
	}

	boost::mpi::broadcast(context.comm, n, 0);
	boost::mpi::broadcast(context.comm, d, 0);

	int numblocks = ((int) n/ (int) blocksize); // of size blocksize
	int leftover = n % blocksize;
	int block = blocksize;

	X.Resize(d, n);
	Y.Resize(n,1);

	for(int i=0; i<numblocks+1; i++) {

	            if (i==numblocks)
	                block = leftover;
	            if (block==0)
	                break;

	            DistCircMatrixType x(d, block), y(block, 1);
	            x.SetRoot(0);
	            y.SetRoot(0);
	            elem::MakeZeros(x);

                if(context.rank==0) {

                    cout << "Reading and distributing chunk " << i*blocksize << " to " << i*blocksize + block - 1 << " ("<< block << " elements )" << endl;
                    double *Xdata = x.Matrix().Buffer();
                    double *Ydata = y.Matrix().Buffer();

                    t = 0;
                    while(!file.eof() && t<block) {
                        getline(file, line);
                        if( line.length()==0) {
                            break;
                        }

                        istringstream tokenstream (line);
                        tokenstream >> label;
                        Ydata[t] = label;

                        while (tokenstream >> token)
                         {
                            delim  = token.find(':');
                            ind = token.substr(0, delim);
                            val = token.substr(delim+1); //.substr(delim+1);
                            j = atoi(ind.c_str()) - 1;
                            Xdata[t * d + j] = atof(val.c_str());
                         }

                        t++;
                    }
                 }

                // The calls below should distribute the data to all the nodes.
               // if (context.rank==0)
                //    cout << "Distributing Data.." << endl;

                elem::DistMatrix<double, elem::STAR, elem::VC> viewX;
                elem::DistMatrix<double, elem::VC, elem::STAR> viewY;

                elem::View(viewX, X, 0, i*blocksize, x.Height(), x.Width());
                elem::View(viewY, Y, i*blocksize, 0, x.Width(), 1);

                viewX = x;
                viewY = y;

//	X = x;
//	Y = y;
	}

	double readtime = timer.elapsed();
	if (context.rank==0)
		cout << "Read Matrix with dimensions: " << n << " by " << d << " (" << readtime << "secs)" << endl;
}

void read_libsvm(skylark_context_t& context, string fName, sparse_matrix_t& X,
                        elem::Matrix<double>& Y,
                        int min_d = 0) {
    if (context.rank==0)
            cout << "Reading sparse matrix from file " << fName << endl;

    boost::mpi::communicator world;

    ifstream file(fName.c_str());
    string line;
    string token, val, ind;
    float label;
    unsigned int start = 0;
    unsigned int delim, t;
    int n = 0;
    int d = 0;
    int i, j, last;
    char c;
    int nnz=0;
    int nz;

    bmpi::timer timer;

    // make one pass over the data to figure out dimensions - will pay in terms of preallocated storage.
    if (context.rank==0) {
        while(!file.eof()) {
            getline(file, line);
            if(line.length()==0)
                break;
            delim = 0;
            for(nz=0;nz<line.length(); nz++) {
                if (line[nz]==':') {
                    nnz++;
                    delim = nz;
                }
            }

            //delim = line.find_last_of(":");
            if(delim > line.length())
                continue;
            n++;
            t = delim;
            while(line[t]!=' ') {
                t--;
            }
            val = line.substr(t+1, delim - t);
            last = atoi(val.c_str());
            if (last>d)
                d = last;
        }
        if (min_d > 0)
            d = std::max(d, min_d);

        // prepare for second pass
        file.clear();
        file.seekg(0, std::ios::beg);
    }

    // rough number of non-zeros per process - as soon as these many nnzs are read, the chunk of the rows read is sent to
    // appropriate process
    int nnzs_per_process;

    if (context.rank==0) {
        nnzs_per_process =  nnz / context.size;
        std::cout << "Total nnz = " << nnz <<  "; nnzs per process = " << nnzs_per_process << std::endl;
    }

    context.comm.barrier();

    for(int r=context.size-1; r>=0; r--) {

        if(context.rank==0) {

            vector<int> col_ptr;
            vector<int> rowind;
            vector<double> values;
            vector<double> y;
            // send sparse matrices
            t = 0;
            int nnz_local = 0;

            while(!file.eof()) {
                getline(file, line);
                t++;
                if( line.length()==0) {
                    break;
                }

                istringstream tokenstream (line);
                tokenstream >> label;
                y.push_back(label);
                col_ptr.push_back(nnz_local);

                while (tokenstream >> token)
                {
                    delim  = token.find(':');
                    ind = token.substr(0, delim);
                    val = token.substr(delim+1); //.substr(delim+1);
                    j = atoi(ind.c_str()) - 1;
                    rowind.push_back(j);
                    values.push_back(atof(val.c_str()));
                    nnz_local++;
                }

                if ((nnz_local>nnzs_per_process) && (r>0)) {
                    std::cout << "Sending to " << r << std::endl;
                    col_ptr.push_back(nnz_local);
                    world.send(r, 1, t);
                    world.send(r, 2, d);
                    world.send(r, 3, nnz_local);
                    world.send(r, 4, &col_ptr[0], col_ptr.size());
                    world.send(r, 5, &rowind[0], col_ptr.size());
                    world.send(r, 6, &values[0], col_ptr.size());
                    world.send(r, 7, &y[0], y.size());
                    break;
                }
            }

            if (r==0) {
                std::cout << "rank=0: Read " << t << " x " << d << " with " << nnz_local << " nonzeros" << std::endl;
                col_ptr.push_back(nnz_local);
                X.attach(&col_ptr[0], &rowind[0], &values[0], t+1, nnz_local, t, d);
                Y.Resize(t,1);
                Y.Attach(t,1,&y[0],0);
                /*world.send(r, 1, t);
                world.send(r, 2, d);
                world.send(r, 3, nnz_local);
                world.send(r, 4, &col_ptr[0], col_ptr.size());
                world.send(r, 5, &rowind[0], col_ptr.size());
                world.send(r, 6, &values[0], col_ptr.size());
                world.send(r, 7, &y[0], y.size());*/
                break;
            }

        } else {

            //receive
        if (context.rank==r) {
                int nnz_local, t, d;
                world.recv(0, 1, t);
                world.recv(0, 2, d);
                world.recv(0, 3, nnz_local);
                std::cout << r << " is receiving from 0: sparse matrix " << t << " x " << d << " " << nnz_local  << " nonzeros )" << std::endl;
                double* values = new double[nnz_local];
                int* rowind = new int[nnz_local];
                int* col_ptr = new int[t+1];

                world.recv(0, 4, col_ptr, t+1);
                world.recv(0, 5, rowind, nnz_local);
                world.recv(0, 6, values, nnz_local);

                X.attach(col_ptr, rowind, values, t+1, nnz_local, t, d);

                double* y = new double[t];
                world.recv(0, 7, y, t);
                Y.Resize(t,1);
                Y.Attach(t,1,y,0);

                std::cout << "Read " << t << " x " << d << " with " << nnz_local << " nonzeros" << std::endl;
            }
        }
    }


    double readtime = timer.elapsed();
    if (context.rank==0)
        cout << "Read Matrix with dimensions: " << n << " by " << d << " (" << readtime << "secs)" << endl;

}


template <class InputType, class LabelType>
void read(skylark::sketch::context_t& context, int fileformat, string filename, InputType& X, LabelType& Y, int d=0) {
    switch(fileformat) {
            case LIBSVM_DENSE: case LIBSVM_SPARSE:
            {
                read_libsvm(context, filename, X, Y, d);
                break;
            }
            case HDF5:
            {
                #ifdef SKYLARK_HAVE_HDF5
                    read_hdf5_dense(context, filename, X, Y, d);
                #else
                    // TODO
                #endif
                break;
            }
        }
}

void read_model_file(string fName, elem::Matrix<double>& W) {
	ifstream file(fName.c_str());
	string line, token;
	string prefix = "# Dimensions";
	int i=0;
	int j;
	int m, n;
	while(!file.eof()) {
			getline(file, line);
			if(line.compare(0, prefix.size(), prefix) == 0) {
				istringstream tokenstream (line.substr(prefix.size(), line.size()));
				tokenstream >> token;

				m = atoi(token.c_str());
				tokenstream >> token;

				n = atoi(token.c_str());
				std::cout << "Read coefficients of size " << m << " x " << n << std::endl;
				W.Resize(m,n);
				continue;
			}
			else {
				if(line[0] == '#' || line.length()==0)
					continue;
			}

			istringstream tokenstream (line);
			j = 0;
			while (tokenstream >> token){
				W.Set(i,j, atof(token.c_str()));
				j++;
			}
			i++;
	}
}

void SaveModel(skylark::sketch::context_t& context, hilbert_options_t& options, elem::Matrix<double> W)  {
    if (context.rank==0) {
            std::stringstream dimensionstring;
            dimensionstring << "# Dimensions " << W.Height() << " " << W.Width() << "\n";
            elem::Write(W, options.modelfile, elem::ASCII, options.print().append(dimensionstring.str()));
        }
}


#endif /* IO_HPP_ */
