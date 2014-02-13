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

using namespace std;
namespace bmpi =  boost::mpi;

typedef skylark::sketch::context_t skylark_context_t;
typedef elem::DistMatrix<double, elem::CIRC, elem::CIRC> DistCircMatrixType;


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
