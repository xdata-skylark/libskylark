#ifndef SKYLARK_ML_MODEL_HPP
#define SKYLARK_ML_MODEL_HPP

#include <El.hpp>
#include <El/core/types.h>
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
#include "options.hpp"

#ifdef SKYLARK_HAVE_OPENMP
#include <omp.h>
#endif

namespace skylark { namespace ml {

int classification_accuracy(El::Matrix<double>& Yt, El::Matrix<double>& Yp) {
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

struct model_t {
    // TODO the following two should depend on the input type
    // TODO explicit doubles is not desired.
    typedef El::Matrix<double> intermediate_type;
    typedef El::Matrix<double> coef_type;

    typedef sketch::sketch_transform_t<boost::any, boost::any>
    feature_transform_type;

    template<typename SketchTransformType>
    model_t(std::vector<const SketchTransformType *>& maps, bool scale_maps,
        int num_features, int num_outputs, bool regression) :
        _coef(num_features, num_outputs), _maps(maps.size()), _scale_maps(scale_maps),
        _regression(regression), _starts(maps.size()), _finishes(maps.size()) {

        // TODO verify all N dimension of the maps match

        El::Zero(_coef);

        int nf = 0;
        for(int i = 0; i < maps.size(); i++) {
            _maps[i] = maps[i]->type_erased();
            _starts[i] = nf;
            _finishes[i] = nf + _maps[i]->get_S() - 1;
            nf += _maps[i]->get_S();
        }

        _input_size = (_maps.size() == 0) ?
            num_features : _maps[0]->get_N();
    }

    model_t(const boost::property_tree::ptree &pt) {
        build_from_ptree(pt);
    }

    model_t(const std::string& fname) {
        std::ifstream is(fname);

        // Skip all lines begining with "#"
        while(is.peek() == '#')
            is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        boost::property_tree::ptree pt;
        boost::property_tree::read_json(is, pt);
        is.close();
        build_from_ptree(pt);
    }

    ~model_t() {
        for (auto it = _maps.begin(); it != _maps.end(); it++)
            delete *it;
    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;
        pt.put("skylark_object_type", "model:linear-on-features");
        pt.put("skylark_version", VERSION);

        pt.put("num_features", _coef.Height());
        pt.put("num_outputs", _coef.Width());
        pt.put("input_size", _input_size);
        pt.put("regression", _regression);

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
        El::Print(_coef, "", scoef);
        pt.put("coef_matrix", scoef.str());

        return pt;
    }

    /**
     * Saves the model to a file named fname. You may want to use this method
     * from only a single rank.
     */
    void save(const std::string& fname, const std::string& header) const {
        boost::property_tree::ptree pt = to_ptree();
        std::ofstream of(fname);
        of << header;
        boost::property_tree::write_json(of, pt);
        of.close();
    }

    template<typename InputType, typename LabelType, typename DecisionType>
    void predict(const InputType& X, LabelType& PV, DecisionType& DV,
        int num_threads = 1) const {

        int d = base::Height(X);
        int k = base::Width(_coef);
        int n = base::Width(X);

        if (_maps.size() == 0)  {
            // No maps (linear case)

            DV.Resize(n, k);
            base::Gemm(El::TRANSPOSE,El::NORMAL,1.0, X, _coef, 0.0, DV);
        } else {
            // Non-linear case
            coef_type Wslice;
            int j, start, finish, sj;

            El::Zeros(DV, n, k);
#           ifdef SKYLARK_HAVE_OPENMP
#           pragma omp parallel for if(num_threads > 1) private(j, start, finish, sj) num_threads(num_threads)
#           endif
            for(j = 0; j < _maps.size(); j++) {
                start = _starts[j];
                finish = _finishes[j];
                sj = finish - start  + 1;

                intermediate_type z(sj, n);
                _maps[j]->apply(&X, &z, sketch::columnwise_tag());

                if (_scale_maps)
                    El::Scale(sqrt(double(sj) / d), z);

                DecisionType o(n, k);

                El::LockedView(Wslice, _coef, start, 0, sj, k);
                base::Gemm(El::TRANSPOSE, El::NORMAL, 1.0, z, Wslice, o);

#               ifdef SKYLARK_HAVE_OPENMP
#               pragma omp critical
#               endif
                base::Axpy(+1.0, o, DV);
            }
        }

        if (!_regression) {
            double o, o1, pred;
            PV.Resize(n, 1);
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
    }

    coef_type& get_coef() { return _coef; }

    int get_output_size() const { return _coef.Width(); }
    int get_input_size() const { return _input_size; }

    bool is_regression() const { return _regression; }

protected:

    void build_from_ptree(const boost::property_tree::ptree &pt) {
        int num_features = pt.get<int>("num_features");
        int num_outputs = pt.get<int>("num_outputs");
        _coef.Resize(num_features, num_outputs);

        _input_size = pt.get<int>("input_size");
        _regression = pt.get<bool>("regression");

        int num_maps = pt.get<int>("feature_mapping.number_maps");
        _maps.resize(num_maps);
        const boost::property_tree::ptree &ptmaps =
            pt.get_child("feature_mapping.maps");
        for(int i = 0; i < num_maps; i++)
            _maps[i] =
                feature_transform_type::from_ptree(
                   ptmaps.get_child(std::to_string(i)));

        int nf = 0;
        _starts.resize(num_maps);
        _finishes.resize(num_maps);
        for(int i = 0; i < _maps.size(); i++) {
            _starts[i] = nf;
            _finishes[i] = nf + _maps[i]->get_S() - 1;
            nf += _maps[i]->get_S();
        }

        _scale_maps = pt.get<bool>("feature_mapping.scale_maps");

        std::istringstream coef_str(pt.get<std::string>("coef_matrix"));
        double *buffer = _coef.Buffer();
        int ldim = _coef.LDim();
        for(int i = 0; i < num_features; i++) {
            std::string line;
            std::getline(coef_str, line);
            std::istringstream coefstream(line);
            for(int j = 0; j < num_outputs; j++) {
                std::string token;
                coefstream >> token;
                buffer[i + j * ldim] = atof(token.c_str());
            }
        }
    }

private:
    coef_type _coef;
    El::Int _input_size, _output_size;
    std::vector<const feature_transform_type *> _maps; // TODO use shared_ptr
    bool _scale_maps;
    bool _regression;

    std::vector<int> _starts, _finishes;
};

template<typename OutType, typename ComputeType = OutType>
struct kernel_model_t {
    // TODO: so far assume only Gaussian kernel.
    typedef ComputeType compute_type;
    typedef OutType out_type;

    kernel_model_t(gaussian_t &k,
        base::direction_t direction, El::DistMatrix<compute_type> &X,
        const std::string &dataloc, const El::DistMatrix<compute_type> &A) :
        _X(X), _direction(direction),
        _A(std::move(A)), _dataloc(dataloc), _k(k),
        _input_size(k.get_dim()), _output_size(A.Width()){

    }

    void predict(base::direction_t direction_XT, 
        const El::DistMatrix<compute_type> &XT, El::DistMatrix<out_type> &YP) const {

        El::DistMatrix<double> KT;
        Gram(_direction, direction_XT, _k, _X, XT, KT);
        El::Gemm(El::ADJOINT, El::NORMAL, 1.0, _A, KT, YP);
    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;

        pt.put("skylark_object_type", "model:kernel");
        pt.put("skylark_version", VERSION);

        pt.put("data_location", _dataloc);
        pt.put("num_outputs", _output_size);
        pt.put("input_size", _input_size);
        pt.put("regression", true);

        pt.add_child("kernel", _k.to_ptree());

        std::stringstream sA;
        El::Print(_A, "", sA);
        pt.put("alpha", sA.str());

        return pt;
    }

    void save(const std::string& fname, const std::string& header) const {
        boost::property_tree::ptree pt = to_ptree();
        std::ofstream of(fname);
        of << header;
        boost::property_tree::write_json(of, pt);
        of.close();
    }

private:
    const El::DistMatrix<compute_type> &_X;
    const base::direction_t _direction;
    const El::DistMatrix<compute_type> &_A;
    const std::string _dataloc;
    const gaussian_t _k;
    const El::Int _input_size, _output_size;
};

} }


#endif /* SKYLARK_ML_MODEL_HPP */
