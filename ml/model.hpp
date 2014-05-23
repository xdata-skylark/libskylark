#ifndef SKYLARK_ML_MODEL_HPP
#define SKYLARK_ML_MODEL_HPP

#include <elemental.hpp>
#include <skylark.hpp>
#include <cmath>
#include <boost/mpi.hpp>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include "kernels.hpp"

#ifdef SKYLARK_HAVE_OPENMP
#include <omp.h>
#endif


typedef elem::Matrix<double> LocalMatrixType;

namespace skylark { namespace ml {

int classification_accuracy(LocalMatrixType& Yt, LocalMatrixType& Yp) {
    int correct = 0;
        double o, o1;
        int pred;


        for(int i=0; i < Yp.Height(); i++) {
            o = Yp.Get(i,0);
            pred = 0;
            if (Yp.Width()==1)
                    pred = (o >= 0)? +1:-1;

            for(int j=1; j < Yp.Width(); j++) {
                o1 = Yp.Get(i,j);
                if ( o1 > o) {
                    o = o1;
                    pred = j;
                }
            }

            if(pred == (int) Yt.Get(i,0))
                correct++;
        }
        return correct;
}


template <class T>
struct Model
{
public:
	typedef skylark::sketch::sketch_transform_t<T, LocalMatrixType>
	    feature_transform_t;
	typedef std::vector<const feature_transform_t *> feature_transform_array_t;

	Model<T>(feature_transform_array_t& featureMaps, bool ScaleFeatureMaps, std::vector<int>& starts, std::vector<int>& finishes, int dimensions, int NumFeatures, int NumTargets);
	Model<T>(std::string fName, const boost::mpi::communicator& comm);
	void predict(T& X, LocalMatrixType& PredictedLabels, LocalMatrixType& DecisionValues);
	void get_probabilities(T& X, LocalMatrixType& Probabilities);
	double evaluate(LocalMatrixType& Yt, LocalMatrixType& Yp, const boost::mpi::communicator& comm);
	void save(std::string fName, std::string header, int rank);
	// void load(std::string fName);
	elem::Matrix<double>& get_coef() {return *Wbar;}
	void set_num_threads(int nthreads) {NumThreads = nthreads;}
	int get_classes() {return classes;}

private:
	 feature_transform_array_t* featureMaps;
	 elem::Matrix<double>* Wbar;
	 std::vector<int>* starts;
	 std::vector<int>* finishes;
	 int NumThreads;
	 bool ScaleFeatureMaps;
	 int NumFeatures;
	 int dimensions;
	 int classes;
};


template <class T>
Model<T>::Model(feature_transform_array_t& featureMaps,  bool ScaleFeatureMaps, std::vector<int>& starts, std::vector<int>& finishes, int dimensions, int NumFeatures, int NumTargets) {
	this->Wbar = new elem::Matrix<double>(NumFeatures, NumTargets);
	elem::MakeZeros(*Wbar);
	this->featureMaps = new feature_transform_array_t();
	*(this->featureMaps) = featureMaps;
	this->starts = new std::vector<int> ();
	*(this->starts) = starts;
	this->finishes = new std::vector<int> ();
	*(this->finishes) = finishes;
	this->NumThreads = 1;
	this->ScaleFeatureMaps = ScaleFeatureMaps;
	this->NumFeatures = NumFeatures;
	this->dimensions = dimensions;
	this->classes = NumTargets;
}

template <class T>
void Model<T>::save(std::string fName, std::string header, int rank) {
	std::stringstream dimensionstring;

	dimensionstring << "# CoefficientDimensions " << Wbar->Height() << " " << Wbar->Width() << "\n" << "# InputDimensions " << dimensions << "\n";
	if (rank==0)
	   elem::Write(*Wbar, fName, elem::ASCII, header.append(dimensionstring.str()));
	}

template <class T>
Model<T>::Model(std::string fName, const boost::mpi::communicator& comm) {
	std::ifstream file(fName.c_str());
	std::string line;
	std::getline(file, line);

	int pos = line.find(":", 0);
	std::string commandline  = line.substr(pos+1, std::string::npos);
	std::cout << "line:" << commandline << std::endl;
	std::istringstream tokenstream (commandline);
	std::string token;
	std::vector<std::string> argvec;
	while (tokenstream >> token) {
		argvec.push_back(token);
	}
	int argc = argvec.size();
	char ** argv = new char*[argvec.size()];
	for(size_t i = 0; i < argvec.size(); i++){
	    argv[i] = new char[argvec[i].size() + 1];
	    strcpy(argv[i], argvec[i].c_str());
	}

	hilbert_options_t options (argc, argv, comm.size());
	skylark::base::context_t context (options.seed);

	std::string tok, inputdimensions, coefwidth, coefheight;
	while(line.substr(0,1) == "#") {
		std::istringstream tokenstream2 (line);
		tokenstream2 >> tok;
		tokenstream2 >> tok;
		if (tok=="InputDimensions") {
			tokenstream2 >> inputdimensions;
		}
		if (tok=="CoefficientDimensions") {
			tokenstream2 >> coefheight;
			tokenstream2 >> coefwidth;
		}

		std::getline(file, line);
	}
	int dimensions = atoi(inputdimensions.c_str());
	int height = atoi(coefheight.c_str());
	int width = atoi(coefwidth.c_str());

	this->dimensions = dimensions;

	Wbar = new elem::Matrix<double>(height, width);
	this->NumFeatures = height;
	this->classes = width;

	skylark::ml::kernels::gaussian_t gaussian(dimensions, options.kernelparam);
	skylark::ml::kernels::polynomial_t poly(dimensions,
			options.kernelparam, options.kernelparam2, options.kernelparam3);
	skylark::ml::kernels::laplacian_t lap(dimensions, options.kernelparam);
	skylark::ml::kernels::expsemigroup_t semigrp(dimensions, options.kernelparam);

	this->featureMaps = new feature_transform_array_t(options.numfeaturepartitions);
	this->starts = new std::vector<int> (options.numfeaturepartitions);
	this->finishes = new std::vector<int> (options.numfeaturepartitions);

	int blksize = int(ceil(double(NumFeatures) / options.numfeaturepartitions));

	    for(int i = 0; i < options.numfeaturepartitions; i++) {


	        (*starts)[i] = i * blksize;
	        (*finishes)[i] = std::min((i + 1) * blksize, NumFeatures) - 1;
	        int sj = (*finishes)[i] - (*starts)[i] + 1;

	        switch(options.kernel) {
				case GAUSSIAN:
					if(options.regularmap)
						(*featureMaps)[i]  = gaussian.template create_rft< T, LocalMatrixType >(sj, skylark::ml::regular_feature_transform_tag(), context);
					else
						(*featureMaps)[i]  = gaussian.template create_rft< T, LocalMatrixType >(sj, skylark::ml::fast_feature_transform_tag(), context);
					break;
				case POLYNOMIAL:
				         (*featureMaps)[i] = poly.template create_rft< T, LocalMatrixType >(sj, skylark::ml::regular_feature_transform_tag(), context);
					break;
				case LAPLACIAN:
				         (*featureMaps)[i] = lap.template create_rft< T, LocalMatrixType >(sj, skylark::ml::regular_feature_transform_tag(), context);
					break;
				case EXPSEMIGROUP:
			 			(*featureMaps)[i]  = semigrp.template create_rft< T, LocalMatrixType >(sj, skylark::ml::regular_feature_transform_tag(), context);
					break;
	        }


	  }

	this->ScaleFeatureMaps = true;

    double* buffer = Wbar->Buffer();
    const int ldim = Wbar->LDim();
    int i = 0;

    std::cout << "Reading coefficients" << std::endl;
    while(!file.eof()) {
    	std::istringstream coefstream (line);
    	int j = 0;
    	while(coefstream >> token) {
    	//	std::cout << " " << i << " " << j << " " << token << std::endl;
    		buffer[i+j*ldim] = atof(token.c_str());
    		j++;
    	}
    	std::getline(file, line);
    	if (j>0)
    		i++;
    }

    this->NumThreads = 1.0;
}


template <class T>
void Model<T>::predict(T& X, LocalMatrixType& PredictedLabels, LocalMatrixType& DecisionValues) {
    // TOD W should be really kept as part of the model

    // int n = X.Width();
    int d = skylark::base::Height(X);
    int k = skylark::base::Width(DecisionValues);
    int ni = skylark::base::Width(X);
    int j, start, finish, sj;
    const feature_transform_t* featureMap;



    if (featureMaps->size() == 0) {
        DecisionValues.Resize(ni, k);
        skylark::base::Gemm(elem::TRANSPOSE,elem::NORMAL,1.0, X, *Wbar, 0.0, DecisionValues);
        return;
    }

    elem::Zeros(DecisionValues, ni, k);

    LocalMatrixType Wslice;

#   ifdef SKYLARK_HAVE_OPENMP
#   pragma omp parallel for if(NumThreads > 1) private(j, start, finish, sj, featureMap) num_threads(NumThreads)
#   endif
    for(j = 0; j < featureMaps->size(); j++) {
        start = (*starts)[j];
        finish = (*finishes)[j];
        sj = finish - start  + 1;



        elem::Matrix<double> z(sj, ni);
        featureMap = (*featureMaps)[j];

        featureMap->apply(X, z, skylark::sketch::columnwise_tag());

        if (ScaleFeatureMaps)
            elem::Scal(sqrt(double(sj) / d), z);


        elem::Matrix<double> o(ni, k);

        elem::View(Wslice, *Wbar, start, 0, sj, k);
        elem::Gemm(elem::TRANSPOSE, elem::NORMAL, 1.0, z, Wslice, 0.0, o);

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp critical
#       endif
        elem::Axpy(+1.0, o, DecisionValues);
    }

    double o, o1, pred;



    for(int i=0; i < DecisionValues.Height(); i++) {
                o = DecisionValues.Get(i,0);
                pred = 0;
                if (DecisionValues.Width()==1)
                        pred = (o >= 0)? +1:-1;

                for(int j=1; j < DecisionValues.Width(); j++) {
                    o1 = DecisionValues.Get(i,j);
                    if ( o1 > o) {
                        o = o1;
                        pred = j;
                    }
                }

           //     if(pred == (int) Yt.Get(i,0))
            //        correct++;

                PredictedLabels.Set(i,0, pred);
      }

}


template <class T>
double Model<T>::evaluate(LocalMatrixType& Yt,
    LocalMatrixType& Yp, const boost::mpi::communicator& comm) {

    int rank = comm.rank();

    int correct = classification_accuracy(Yt, Yp);
    double accuracy = 0.0;
    int totalcorrect, total;
    boost::mpi::reduce(comm, correct, totalcorrect, std::plus<double>(), 0);
    boost::mpi::reduce(comm, Yt.Height(), total, std::plus<int>(), 0);

    if(rank ==0)
        accuracy =  totalcorrect*100.0/total;
    return accuracy;
}


} }


#endif /* SKYLARK_ML_MODEL_HPP */
