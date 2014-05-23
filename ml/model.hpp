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

int classification_accuracy(elem::Matrix<double>& Yt, elem::Matrix<double>& Yp) {
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

template <typename InputType, typename OutputType>
struct model_t
{
public:
    typedef InputType input_type;
    typedef OutputType output_type;

    // TODO the following two should depend on the input type
    // TODO explicit doubles is not desired.
    typedef elem::Matrix<double> intermediate_type;
    typedef elem::Matrix<double> coef_type;

    typedef skylark::sketch::sketch_transform_t<input_type, intermediate_type>
    feature_transform_type;

    model_t(std::vector<const feature_transform_type *>& maps, bool scale_maps,
        int num_features, int num_outputs) :
        _coef(num_features, num_outputs), _maps(maps), _scale_maps(scale_maps),
        _starts(maps.size()), _finishes(maps.size()) {

        elem::MakeZeros(_coef);

        int nf = 0;
        for(int i = 0; i < _maps.size(); i++) {
            _starts[i] = nf;
            _finishes[i] = nf + _maps[i]->get_S() - 1;
            nf += _maps[i]->get_S();
        }
    }

    model_t(const boost::property_tree::ptree &pt) {
        // TODO
    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        pt.put("skylark_object_type", "model:linear-on-features");
        pt.put("skylark_version", VERSION);

        pt.put("num_features", _coef.Height());
        pt.put("num_outputs", _coef.Width());

        boost::property_tree::ptree ptfmap;
        ptfmap.put("number_maps", _maps.size());
        ptfmap.put("scale_maps", _scale_maps);

        boost::property_tree::ptree ptmaps;
        for(int i = 0; i < _maps.size(); i++)
            ptmaps.push_back(std::make_pair(std::to_string(i),
                    _maps[i]->to_ptree()));
        ptfmap.add_child("maps", ptmaps);

        pt.add_child("feature_mapping", ptfmap);

        std::stringstream scoef;
        elem::Print(_coef, "", scoef);
        pt.put("coef_matrix", scoef.str());

        return pt;
    }

    void predict(input_type& X, output_type& PV, output_type& DV,
        int num_threads = 1) {

        int d = base::Height(X);
        int k = base::Width(_coef);
        int n = base::Width(X);

        if (_maps.size() == 0)  {
            DV.Resize(n, k);
            base::Gemm(elem::TRANSPOSE,elem::NORMAL,1.0, X, _coef, 0.0, DV);
            return;
        }

        coef_type Wslice;
        int j, start, finish, sj;

        elem::Zeros(DV, n, k);
#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for if(num_threads > 1) private(j, start, finish, sj) num_threads(num_threads)
#       endif
        for(j = 0; j < _maps.size(); j++) {
            start = _starts[j];
            finish = _finishes[j];
            sj = finish - start  + 1;

            intermediate_type z(sj, n);
            _maps[j]->apply(X, z, sketch::columnwise_tag());

            if (_scale_maps)
                elem::Scal(sqrt(double(sj) / d), z);

            output_type o(n, k);

            elem::View(Wslice, _coef, start, 0, sj, k);
            base::Gemm(elem::TRANSPOSE, elem::NORMAL, 1.0, z, Wslice, o);

#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp critical
#           endif
            base::Axpy(+1.0, o, DV);
        }

        double o, o1, pred;
        for(int i=0; i < DV.Height(); i++) {
            o = DV.Get(i,0);
            pred = 0;
            if (DV.Width()==1)
                pred = (o >= 0)? +1:-1;

            for(int j=1; j < DV.Width(); j++) {
                o1 = DV.Get(i,j);
                if ( o1 > o) {
                    o = o1;
                    pred = j;
                }
            }

            PV.Set(i,0, pred);
        }
    }

    void get_probabilities(input_type& X, output_type& P, int num_threads = 1);
    coef_type& get_coef() { return _coef; }
    static double evaluate(output_type& Yt, output_type& Yp,
        const boost::mpi::communicator& comm);

    int get_num_outputs() const { return _coef.Width(); }

private:
    coef_type _coef;
    std::vector<const feature_transform_type *> _maps; // TODO use shared_ptr
    bool _scale_maps;

    std::vector<int> _starts, _finishes;
};

template <typename InputType, typename OutputType>
double model_t<InputType, OutputType>::evaluate(OutputType& Yt,
    OutputType& Yp, const boost::mpi::communicator& comm) {

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
