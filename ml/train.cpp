#include <elemental.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <string>
#include <boost/mpi.hpp>
#include "../sketch/context.hpp"

namespace bmpi =  boost::mpi;
using namespace std;

typedef skylark::sketch::context_t skylark_context_t;
typedef elem::DistMatrix<float, elem::distribution_wrapper::VC, elem::distribution_wrapper::STAR> DistInputMatrixType;
typedef elem::DistMatrix<float, elem::distribution_wrapper::CIRC, elem::distribution_wrapper::CIRC> DistCircMatrixType;

void print_vector(vector<float> x) {
	for( std::vector<float>::const_iterator i = x.begin(); i != x.end(); ++i)
	    cout << *i << " ";
}
void read_libsvm_dense(skylark_context_t& context, string fName, DistInputMatrixType& X, DistInputMatrixType& Y) {
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


	// make one pass over the data to figure out dimensions - will pay in terms of preallocated storage.
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

	DistCircMatrixType x(n, d), y(n, 1);
	x.SetRoot(0);
	y.SetRoot(0);

	if(context.rank==0) {
		float *Xdata = x.Matrix().Buffer();
		float *Ydata = y.Matrix().Buffer();

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
				Xdata[d*i + j] = atof(val.c_str());
			 }
		}
	}

	// The calls below should distribute the data to all the nodes.
	cout << "Distributing Data.." << endl;

	X = x;
	Y = y;

	cout << "Read Matrix with dimensions: " << n << " by " << d << endl;
}

int main (int argc, char** argv) {
	/* Initialize MPI */
	  bmpi::environment env (argc, argv);

	/* Create a global communicator */
	  bmpi::communicator world;

	 skylark::sketch::context_t context (12345, world);

	 elem::Initialize (argc, argv);
	 MPI_Comm mpi_world(world);
	 //elem::Grid grid (mpi_world);

	 DistInputMatrixType X, Y;

	 read_libsvm_dense(context, string(argv[1]), X, Y);

	 cout << " Rank " << context.rank << " owns : " << X.LocalHeight() <<  " x " << X.LocalWidth() << endl;

	 return 0;
}
