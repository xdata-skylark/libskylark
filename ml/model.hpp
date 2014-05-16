#ifndef SKYLARK_ML_MODEL_HPP
#define SKYLARK_ML_MODEL_HPP

#include <elemental.hpp>
#include <skylark.hpp>
#include <cmath>
#include <boost/mpi.hpp>
#include <string>

#ifdef SKYLARK_HAVE_OPENMP
#include <omp.h>
#endif


typedef elem::Matrix<double> LocalMatrixType;

namespace skylark { namespace ml {

template <class T>
struct Model
{
public:
	typedef skylark::sketch::sketch_transform_t<T, LocalMatrixType>
	    feature_transform_t;
	typedef std::vector<const feature_transform_t *> feature_transform_array_t;

	Model<T>(feature_transform_array_t& featureMaps, int NumFeatures, int NumTargets);

	void predict(T& X, LocalMatrixType& Y, LocalMatrixType& Outputs);
	void predict(T& X, LocalMatrixType& Y, LocalMatrixType& Outputs, LocalMatrixType& Prob);
	void save(std::string fName, std::string header, int rank);
	void load(std::string fName);
	elem::Matrix<double>& get_coef() {return *Wbar;}

private:
	 feature_transform_array_t featureMaps;
	 elem::Matrix<double>* Wbar;
};


template <class T>
Model<T>::Model(feature_transform_array_t& featureMaps, int NumFeatures, int NumTargets) {
	this->Wbar = new elem::Matrix<double>(NumFeatures, NumTargets);
	elem::MakeZeros(*Wbar);
	this->featureMaps = featureMaps;
}

template <class T>
void Model<T>::save(std::string fName, std::string header, int rank) {
	std::stringstream dimensionstring;
	dimensionstring << "# Dimensions " << Wbar->Height() << " " << Wbar->Width() << "\n";
	if (rank==0)
	   elem::Write(*Wbar, fName, elem::ASCII, header.append(dimensionstring.str()));
	}

} }


#endif /* SKYLARK_ML_MODEL_HPP */
