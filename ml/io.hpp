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


using namespace std;
namespace bmpi =  boost::mpi;

typedef skylark::sketch::context_t skylark_context_t;
typedef elem::DistMatrix<double, elem::CIRC, elem::CIRC> DistCircMatrixType;


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
        elem::DistMatrix<double, elem::STAR, elem::VC>& X,
        elem::DistMatrix<double, elem::VC, elem::STAR>& Y) {

        bmpi::timer timer;

       H5::H5File file( fName, H5F_ACC_RDONLY );
       H5::DataSet datasetX = file.openDataSet( "X" );
       H5::DataSpace filespaceX = datasetX.getSpace();
       int rank = filespaceX.getSimpleExtentNdims();
       hsize_t dimsX[2]; // dataset dimensions
       rank = filespaceX.getSimpleExtentDims( dimsX );
       hsize_t n = dimsX[0];
       hsize_t d = dimsX[1];

        DistCircMatrixType x(d, n), y(n, 1);
        x.SetRoot(0);
        y.SetRoot(0);
        elem::MakeZeros(x);

        H5::DataSet datasetY = file.openDataSet( "Y" );
        H5::DataSpace filespaceY = datasetY.getSpace();
        hsize_t dimsY[1]; // dataset dimensions
        rank = filespaceX.getSimpleExtentDims( dimsY );


        if(context.rank==0) {
            double *Xdata = x.Matrix().Buffer();
            double *Ydata = y.Matrix().Buffer();

            H5::DataSpace mspace1(2, dimsX);
            datasetX.read( Xdata, H5::PredType::NATIVE_DOUBLE, mspace1, filespaceX );

            H5::DataSpace mspace2(1,dimsY);
            datasetY.read( Ydata, H5::PredType::NATIVE_DOUBLE, mspace2, filespaceY );
        }
        X = x;
        Y = y;

        double readtime = timer.elapsed();
        if (context.rank==0)
                cout << "Read Matrix with dimensions: " << n << " by " << d << " (" << readtime << "secs)" << endl;

}
#endif

void read_libsvm_dense(skylark_context_t& context, string fName,
		elem::DistMatrix<double, elem::STAR, elem::VC>& X, 
		elem::DistMatrix<double, elem::VC, elem::STAR>& Y, 
		int min_d = 0) {
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
	}

	boost::mpi::broadcast(context.comm, n, 0);
	boost::mpi::broadcast(context.comm, d, 0);

	DistCircMatrixType x(d, n), y(n, 1);
	x.SetRoot(0);
	y.SetRoot(0);
	elem::MakeZeros(x);

	if(context.rank==0) {
		double *Xdata = x.Matrix().Buffer();
		double *Ydata = y.Matrix().Buffer();

		// second pass
		file.clear();
		file.seekg(0, std::ios::beg);
		i = -1;
		while(!file.eof()) {
			getline(file, line);
			if( line.length()==0) {
				break;
			}
			i++;
			istringstream tokenstream (line);
			tokenstream >> label;
			Ydata[i] = label;

			while (tokenstream >> token)
			 {
				delim  = token.find(':');
				ind = token.substr(0, delim);
				val = token.substr(delim+1); //.substr(delim+1);
				j = atoi(ind.c_str()) - 1;
				Xdata[i * d + j] = atof(val.c_str());
			 }
		}
	}

	// The calls below should distribute the data to all the nodes.
	if (context.rank==0)
		cout << "Distributing Data.." << endl;

	X = x;
	Y = y;

	double readtime = timer.elapsed();
	if (context.rank==0)
		cout << "Read Matrix with dimensions: " << n << " by " << d << " (" << readtime << "secs)" << endl;

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
				W.ResizeTo(m,n);
				continue;
			}
			else {
				if(line[0] == '#' | line.length()==0)
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

#endif /* IO_HPP_ */
