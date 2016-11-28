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
#include "kernels/kernels.hpp"
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

struct hilbert_model_t {
    // TODO the following two should depend on the input type
    // TODO explicit doubles is not desired.
    typedef El::Matrix<double> intermediate_type;
    typedef El::Matrix<double> coef_type;

    typedef sketch::sketch_transform_t<boost::any, boost::any>
    feature_transform_type;

    template<typename SketchTransformType>
    hilbert_model_t(std::vector<const SketchTransformType *>& maps, bool scale_maps,
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

    hilbert_model_t(const boost::property_tree::ptree &pt) {
        build_from_ptree(pt);
    }

    hilbert_model_t(const std::string& fname) {
        std::ifstream is(fname);

        // Skip all lines begining with "#"
        while(is.peek() == '#')
            is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        boost::property_tree::ptree pt;
        boost::property_tree::read_json(is, pt);
        is.close();
        build_from_ptree(pt);
    }

    ~hilbert_model_t() {
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
                    // TODO shouldn't it be s instead of d?
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
    El::Int _input_size;
    std::vector<const feature_transform_type *> _maps; // TODO use shared_ptr
    bool _scale_maps;
    bool _regression;

    std::vector<int> _starts, _finishes;
};

//////////////////////////////////////////////////////////////////////////


template<typename OutType, typename ComputeType, typename dummy = OutType>
struct model_t;

/**
 * Generic (abstract) model for continous output - regression
 */
template<typename OutType, typename ComputeType>
struct model_t<OutType, ComputeType,
 typename std::enable_if<std::is_floating_point<OutType>::value, OutType>::type>
{

    virtual void predict(base::direction_t direction_XT,
        const El::DistMatrix<ComputeType> &XT, El::DistMatrix<OutType> &YP)
        const = 0;

    virtual boost::property_tree::ptree to_ptree() const = 0;

    virtual void save(const std::string& fname, const std::string& header)
        const = 0;

    virtual El::Int get_input_size() const = 0;

    virtual ~model_t() {

    }
};

/**
 * Generic (abstract) model for discrete (all other) output - classification
 */
template<typename OutType, typename ComputeType>
struct model_t<OutType, ComputeType,
 typename std::enable_if<!std::is_floating_point<OutType>::value, OutType>::type>
{

    virtual void predict(base::direction_t direction_XT,
        const El::DistMatrix<ComputeType> &XT, El::DistMatrix<OutType> &LP,
        El::DistMatrix<ComputeType> &DV)
        const = 0;

    virtual boost::property_tree::ptree to_ptree() const = 0;

    virtual void save(const std::string& fname, const std::string& header)
        const = 0;

    virtual El::Int get_input_size() const = 0;

    virtual void get_column_coding(std::vector<OutType> &rcoding) const = 0;

    virtual ~model_t() {

    }
};

/******************************************************************************/

/**
 * Kernel model.
 */
template<typename KernelType, typename OutType, typename ComputeType = OutType,
         typename dummy = OutType>
struct kernel_model_t;

/**
 * Kernel model for continuous output - regression model
 */
template<typename KernelType, typename OutType, typename ComputeType>
struct kernel_model_t<KernelType, OutType, ComputeType,
  typename std::enable_if<std::is_floating_point<OutType>::value, OutType>::type > :
public model_t<OutType, ComputeType>

{
    typedef KernelType kernel_type;
    typedef ComputeType compute_type;
    typedef OutType out_type;

    kernel_model_t(const kernel_type &k,
        base::direction_t direction, const El::DistMatrix<compute_type> &X,
        const std::string &dataloc, El::Int partial,
        const utility::io::fileformat_t fileformat,
        sketch::generic_sketch_container_t pretransform,
        const El::DistMatrix<compute_type> &A) :
        _X(), _direction(direction),
        _A(), _dataloc(dataloc), _partial(partial),
        _fileformat(fileformat), _pretransform(pretransform), _k(k),
        _input_size(k.get_dim()), _output_size(A.Width()) {

        El::LockedView(_X, X);
        El::LockedView(_A, A);
    }

    kernel_model_t(const boost::property_tree::ptree &pt) {
        build_from_ptree(pt);
    }

    void predict(base::direction_t direction_XT,
        const El::DistMatrix<compute_type> &XT, El::DistMatrix<out_type> &YP) const {

        El::DistMatrix<compute_type> KT;
        Gram(_direction, direction_XT, _k, _X, XT, KT);
        YP.Resize(_A.Height(), KT.Width());
        El::Gemm(El::ADJOINT, El::NORMAL, out_type(1.0), _A, KT, YP);
    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;

        pt.put("skylark_object_type", "model:kernel");
        pt.put("skylark_version", VERSION);

        pt.put("data_location", _dataloc);
        pt.put("partial", _partial);
        pt.put("fileformat", _fileformat);
        if (!_pretransform.empty())
            pt.add_child("pre_transform", _pretransform.to_ptree());
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

    virtual ~kernel_model_t() {

    }

    El::Int get_input_size() const {
        return _input_size;
    }

protected:
    void build_from_ptree(const boost::property_tree::ptree &pt) {

        _input_size = pt.get<El::Int>("input_size");
        _output_size = pt.get<El::Int>("num_outputs");

        _k = kernel_container_t(pt.get_child("kernel"));
        _dataloc = pt.get<std::string>("data_location");
        _fileformat = (utility::io::fileformat_t)pt.get<int>("fileformat");
        _partial = pt.get<El::Int>("partial");

        // TODO handle "partial" and "sampling"
        El::DistMatrix<OutType> dummyY;

        switch (_fileformat) {
        case utility::io::FORMAT_LIBSVM:
            utility::io::ReadLIBSVM(_dataloc, _X0, dummyY, base::COLUMNS,
                _input_size, _partial);
            break;

#ifdef SKYLARK_HAVE_HDF5
        case utility::io::FORMAT_HDF5: {
            H5::H5File in(_dataloc, H5F_ACC_RDONLY);
            utility::io::ReadHDF5(in, "X", _X0, -1, _partial);
            in.close();
        }
            break;
#endif

        default:
            // TODO
            return;
        }

       if (pt.count("pre_transform") > 0) {
            _pretransform =
                sketch::generic_sketch_container_t::from_ptree(pt.
                    get_child("pre_transform"));

            _X.Resize(_X0.Height(), _pretransform.get_S());
            _pretransform.apply(&_X0, &_X, sketch::rowwise_tag());
            _X0.Empty();
        } else
            El::View(_X, _X0);

        _direction = base::COLUMNS;

        std::istringstream A_str(pt.get<std::string>("alpha"));
        if (_A.Grid().Size() == 1) {
            _A.Resize(_X.Width(), _output_size);
            compute_type *buffer = _A.Buffer();
            int ldim = _A.LDim();
            for(int i = 0; i < _X.Width(); i++) {
                std::string line;
                std::getline(A_str, line);
                std::istringstream Astream(line);
                for(int j = 0; j < _output_size; j++) {
                    std::string token;
                    Astream >> token;
                    buffer[i + j * ldim] = atof(token.c_str());
                }
            }
        } else {
            // TODO: can do more memory efficient
            El::DistMatrix<compute_type, El::CIRC, El::CIRC> A0(_X.Width(),
                _output_size);
            if (A0.Grid().Rank() == 0) {
                compute_type *buffer = A0.Buffer();
                int ldim = A0.LDim();
                for(int i = 0; i < _X.Width(); i++) {
                    std::string line;
                    std::getline(A_str, line);
                    std::istringstream Astream(line);
                    for(int j = 0; j < _output_size; j++) {
                        std::string token;
                        Astream >> token;
                        buffer[i + j * ldim] = atof(token.c_str());
                    }
                }
            }

            _A = A0;
        }
    }

private:
    El::DistMatrix<compute_type> _X, _X0;  // X0 is only for load.
    base::direction_t _direction;
    El::DistMatrix<compute_type> _A;
    std::string _dataloc;
    El::Int _partial;
    utility::io::fileformat_t _fileformat;
    sketch::generic_sketch_container_t _pretransform;
    kernel_type _k;
    El::Int _input_size, _output_size;
};

/**
 * Kernel model for discrete (all other) outputs - classification
 */
template<typename KernelType, typename OutType, typename ComputeType>
struct kernel_model_t<KernelType, OutType, ComputeType,
  typename std::enable_if<!std::is_floating_point<OutType>::value, OutType>::type > :
public model_t<OutType, ComputeType> {

    typedef KernelType kernel_type;
    typedef ComputeType compute_type;
    typedef OutType out_type;

    kernel_model_t(const kernel_type &k,
        base::direction_t direction, const El::DistMatrix<compute_type> &X,
        const std::string &dataloc, El::Int partial,
        const utility::io::fileformat_t fileformat,
        const skylark::sketch::generic_sketch_container_t pretransform,
        const El::DistMatrix<compute_type> &A,
        const std::vector<OutType> &rcoding) :
        _X(), _direction(direction),
        _A(), _rcoding(rcoding), _dataloc(dataloc), _partial(partial),
        _fileformat(fileformat), _pretransform(pretransform),
        _k(k), _input_size(k.get_dim()), _output_size(A.Width()) {

        El::LockedView(_X, X);
        El::LockedView(_A, A);
    }

    kernel_model_t(const boost::property_tree::ptree &pt) {
        build_from_ptree(pt);
    }

    void predict(base::direction_t direction_XT,
        const El::DistMatrix<compute_type> &XT, El::DistMatrix<out_type> &LP,
        El::DistMatrix<compute_type> &DV) const {

        El::DistMatrix<compute_type> KT;
        Gram(_direction, direction_XT, _k, _X, XT, KT);
        El::Gemm(El::ADJOINT, El::NORMAL, compute_type(1.0), _A, KT, DV);
        DummyDecode(El::ADJOINT, DV, LP, _rcoding);
    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;

        pt.put("skylark_object_type", "model:kernel");
        pt.put("skylark_version", VERSION);

        pt.put("data_location", _dataloc);
        pt.put("partial", _partial);
        pt.put("fileformat", _fileformat);
        if (!_pretransform.empty())
            pt.add_child("pre_transform", _pretransform.to_ptree());
        pt.put("num_outputs", _output_size);
        pt.put("input_size", _input_size);
        pt.put("regression", false);

        boost::property_tree::ptree rcoding;
        for(int i = 0; i < _rcoding.size(); i++)
            rcoding.put(std::to_string(i), _rcoding[i]);
        pt.add_child("rcoding", rcoding);

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

    virtual ~kernel_model_t() {

    }

    El::Int get_input_size() const {
        return _input_size;
    }

    void get_column_coding(std::vector<OutType> &rcoding) const {
        rcoding.resize(_rcoding.size());
        for(int i = 0; i < _rcoding.size(); i++)
            rcoding[i] = _rcoding[i];
    }

protected:

    void build_from_ptree(const boost::property_tree::ptree &pt) {

        _input_size = pt.get<El::Int>("input_size");
        _output_size = pt.get<El::Int>("num_outputs");
        _rcoding.resize(_output_size);
        const boost::property_tree::ptree &ptrcoding =
            pt.get_child("rcoding");
        for(El::Int i = 0; i < _output_size; i++)
            _rcoding[i] = ptrcoding.get<OutType>(std::to_string(i));
        _k = kernel_container_t(pt.get_child("kernel"));

        _dataloc = pt.get<std::string>("data_location");
        _fileformat = (utility::io::fileformat_t)pt.get<int>("fileformat");
        _partial = pt.get<int>("partial");

        El::DistMatrix<OutType> dummyL;

        switch (_fileformat) {
        case utility::io::FORMAT_LIBSVM:
            utility::io::ReadLIBSVM(_dataloc, _X0, dummyL, base::COLUMNS,
                _input_size, _partial);
            break;

#ifdef SKYLARK_HAVE_HDF5
        case utility::io::FORMAT_HDF5: {
            H5::H5File in(_dataloc, H5F_ACC_RDONLY);
            utility::io::ReadHDF5(in, "X", _X0, -1, _partial);
            in.close();
        }
            break;
#endif

        default:
            // TODO
            return;
        }

        if (pt.count("pre_transform") > 0) {
            _pretransform =
                sketch::generic_sketch_container_t::from_ptree(pt.
                    get_child("pre_transform"));

            _X.Resize(_X0.Height(), _pretransform.get_S());
            _pretransform.apply(&_X0, &_X, sketch::rowwise_tag());
            _X0.Empty();
        } else
            El::View(_X, _X0);

        _direction = base::COLUMNS;

        std::istringstream A_str(pt.get<std::string>("alpha"));
        if (_A.Grid().Size() == 1) {
            _A.Resize(_X.Width(), _output_size);
            compute_type *buffer = _A.Buffer();
            int ldim = _A.LDim();
            for(int i = 0; i < _X.Width(); i++) {
                std::string line;
                std::getline(A_str, line);
                std::istringstream Astream(line);
                for(int j = 0; j < _output_size; j++) {
                    std::string token;
                    Astream >> token;
                    buffer[i + j * ldim] = atof(token.c_str());
                }
            }
        } else {
            // TODO: can do more memory efficient
            El::DistMatrix<compute_type, El::CIRC, El::CIRC> A0(_X.Width(),
                _output_size);
            if (A0.Grid().Rank() == 0) {
                compute_type *buffer = A0.Buffer();
                int ldim = A0.LDim();
                for(int i = 0; i < _X.Width(); i++) {
                    std::string line;
                    std::getline(A_str, line);
                    std::istringstream Astream(line);
                    for(int j = 0; j < _output_size; j++) {
                        std::string token;
                        Astream >> token;
                        buffer[i + j * ldim] = atof(token.c_str());
                    }
                }
            }

            _A = A0;
        }
    }

private:
    El::DistMatrix<compute_type> _X, _X0; // X0 is only for model load.
    base::direction_t _direction;
    El::DistMatrix<compute_type> _A;
    std::vector<OutType> _rcoding;
    std::string _dataloc;
    El::Int _partial;
    utility::io::fileformat_t _fileformat;
    sketch::generic_sketch_container_t _pretransform;
    kernel_type _k;
    El::Int _input_size, _output_size;
};

/******************************************************************************/

/**
 * Feature expansion model - expands feature expansion and then uses
 * linear combination.
 */
template<template <typename, typename> class SketchType,
         typename OutType, typename ComputeType = OutType,
         typename dummy = OutType>
struct feature_expansion_model_t;

/**
 * Feature expansion model for continuous output - regression model
 */
template<template <typename, typename> class SketchType,
         typename OutType, typename ComputeType>
struct feature_expansion_model_t<SketchType, OutType, ComputeType,
  typename std::enable_if<std::is_floating_point<OutType>::value, OutType>::type > :
public model_t<OutType, ComputeType>

{
    typedef ComputeType compute_type;
    typedef OutType out_type;

    typedef SketchType<El::DistMatrix<compute_type>,
                       El::DistMatrix<compute_type> > sketch_type;

    feature_expansion_model_t(const sketch_type &S,
        const El::DistMatrix<compute_type> &W) :
        _W(),  _scale_maps(false), _feature_transforms(1),
        _input_size(S.get_N()), _output_size(W.Width()),
        _feature_size(S.get_S()) {

       _feature_transforms[0] = S;
        El::LockedView(_W, W);
    }

    feature_expansion_model_t(bool scale_maps,
        const std::vector<sketch_type> &transforms,
        const El::DistMatrix<compute_type> &W) :
        _W(), _scale_maps(scale_maps),
        _feature_transforms(transforms),
        _input_size(_feature_transforms[0].get_N()), _output_size(W.Width()),
        _feature_size(0) {

        for(auto it = _feature_transforms.begin();
            it != _feature_transforms.end(); it++)
            _feature_size += it->get_S();
        El::LockedView(_W, W);
    }

    feature_expansion_model_t(const boost::property_tree::ptree &pt) {
        build_from_ptree(pt);
    }

    void predict(base::direction_t direction_XT,
        const El::DistMatrix<compute_type> &XT, El::DistMatrix<out_type> &YP) const {

        if (direction_XT == base::COLUMNS) {
            El::Zeros(YP, _output_size, XT.Width());
            El::DistMatrix<compute_type> ZT, VW;
            El::Int starts = 0;
            for(int i = 0; i < _feature_transforms.size(); i++) {
                const sketch_type &S = _feature_transforms[i];
                ZT.Resize(S.get_S(), XT.Width());
                S.apply(XT, ZT, sketch::columnwise_tag());
                if (_scale_maps)
                    El::Scale<compute_type, compute_type>
                        (sqrt(double(S.get_S()) / _feature_size), ZT);
                base::RowView(VW, _W, starts, S.get_S());
                starts += S.get_S();
                El::Gemm(El::ADJOINT, El::NORMAL, compute_type(1.0), VW, ZT,
                    compute_type(1.0), YP);
            }

        } else {
            El::Zeros(YP, XT.Height(), _output_size);
            El::DistMatrix<compute_type> ZT, VW;
            El::Int starts = 0;
            for(int i = 0; i < _feature_transforms.size(); i++) {
                const sketch_type &S = _feature_transforms[i];
                S.apply(XT, ZT, sketch::rowwise_tag());
                if (_scale_maps)
                    El::Scale<compute_type, compute_type>
                        (sqrt(double(S.get_S()) / _feature_size), ZT);
                base::RowView(VW, _W, starts, S.get_S());
                starts += S.get_S();
                El::Gemm(El::NORMAL, El::NORMAL, compute_type(1.0), ZT, VW,
                    compute_type(1.0), YP);
            }
        }
    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;

        pt.put("skylark_object_type", "model:feature_expansion");
        pt.put("skylark_version", VERSION);

        pt.put("num_outputs", _output_size);
        pt.put("input_size", _input_size);
        pt.put("regression", true);

        boost::property_tree::ptree ptfmap;
        ptfmap.put("number_transforms", _feature_transforms.size());
        ptfmap.put("scale_maps", _scale_maps);

        boost::property_tree::ptree ptmaps;
        for(El::Int i = 0; i < _feature_transforms.size(); i++)
            ptmaps.push_back(std::make_pair(std::to_string(i),
                    _feature_transforms[i].to_ptree()));
        ptfmap.add_child("transforms", ptmaps);

        pt.add_child("feature_mapping", ptfmap);

        std::stringstream sW;
        El::Print(_W, "", sW);
        pt.put("weights", sW.str());

        return pt;
    }

    void save(const std::string& fname, const std::string& header) const {
        boost::property_tree::ptree pt = to_ptree();
        std::ofstream of(fname);
        of << header;
        boost::property_tree::write_json(of, pt);
        of.close();
    }

    virtual ~feature_expansion_model_t() {

    }

    El::Int get_input_size() const {
        return _input_size;
    }

protected:
    void build_from_ptree(const boost::property_tree::ptree &pt) {

        _input_size = pt.get<El::Int>("input_size");
        _output_size = pt.get<El::Int>("num_outputs");

        int num_transforms = pt.get<int>("feature_mapping.number_transforms");
        _scale_maps = pt.get<bool>("feature_mapping.scale_maps");

        El::Int s = 0;
        _feature_transforms.resize(num_transforms);
        const boost::property_tree::ptree &ptmaps =
            pt.get_child("feature_mapping.transforms");
        for(int i = 0; i < num_transforms; i++) {
            _feature_transforms[i] =
                sketch_type(sketch_type::from_ptree(ptmaps.get_child(std::to_string(i))));
            s += _feature_transforms[i].get_S();
        }

        std::istringstream W_str(pt.get<std::string>("weights"));
        if (_W.Grid().Size() == 1) {
            _W.Resize(s, _output_size);
            compute_type *buffer = _W.Buffer();
            int ldim = _W.LDim();
            for(int i = 0; i < s; i++) {
                std::string line;
                std::getline(W_str, line);
                std::istringstream Wstream(line);
                for(int j = 0; j < _output_size; j++) {
                    std::string token;
                    Wstream >> token;
                    buffer[i + j * ldim] = atof(token.c_str());
                }
            }
        } else {
            // TODO: can do more memory efficient
            El::DistMatrix<compute_type, El::CIRC, El::CIRC> W0(s, _output_size);
            if (W0.Grid().Rank() == 0) {
                compute_type *buffer = W0.Buffer();
                int ldim = W0.LDim();
                for(int i = 0; i < s; i++) {
                    std::string line;
                    std::getline(W_str, line);
                    std::istringstream Wstream(line);
                    for(int j = 0; j < _output_size; j++) {
                        std::string token;
                        Wstream >> token;
                        buffer[i + j * ldim] = atof(token.c_str());
                    }
                }
            }

            _W = W0;
        }
    }

private:
    El::DistMatrix<compute_type> _W;
    bool _scale_maps;
    std::vector<sketch_type> _feature_transforms;
    El::Int _input_size, _output_size;
    El::Int _feature_size;
};

/**
 * Approximate kernel model for discrete (all other) outputs - classification
 */
template<template <typename, typename> class SketchType,
         typename OutType, typename ComputeType>
struct feature_expansion_model_t<SketchType, OutType, ComputeType,
  typename std::enable_if<!std::is_floating_point<OutType>::value, OutType>::type > :
public model_t<OutType, ComputeType> {

    typedef ComputeType compute_type;
    typedef OutType out_type;

    typedef SketchType<El::DistMatrix<compute_type>,
                       El::DistMatrix<compute_type> > sketch_type;

    feature_expansion_model_t(const sketch_type &S,
        const El::DistMatrix<compute_type> &W,
        const std::vector<OutType> &rcoding) :
        _W(), _rcoding(rcoding), _scale_maps(false), _feature_transforms(1),
        _input_size(S.get_N()), _output_size(W.Width()),
        _feature_size(S.get_S()) {

        _feature_transforms[0] = S;
        El::LockedView(_W, W);
    }

    feature_expansion_model_t(bool scale_maps,
        const std::vector<sketch_type> &transforms,
        const El::DistMatrix<compute_type> &W,
        const std::vector<OutType> &rcoding) :
        _W(), _rcoding(rcoding), _scale_maps(scale_maps),
        _feature_transforms(transforms),
        _input_size(_feature_transforms[0].get_N()), _output_size(W.Width()),
        _feature_size(0) {

        for(auto it = _feature_transforms.begin();
            it != _feature_transforms.end(); it++)
            _feature_size += it->get_S();
        El::LockedView(_W, W);
    }

    feature_expansion_model_t(const boost::property_tree::ptree &pt) {
        build_from_ptree(pt);
    }

    void predict(base::direction_t direction_XT,
        const El::DistMatrix<compute_type> &XT, El::DistMatrix<out_type> &LP,
        El::DistMatrix<compute_type> &DV) const {

        if (direction_XT == base::COLUMNS) {

            El::DistMatrix<compute_type> ZT, VW;
            El::Zeros(DV, _output_size, XT.Width());
            El::Int starts = 0;
            for(int i = 0; i < _feature_transforms.size(); i++) {
                const sketch_type &S = _feature_transforms[i];
                ZT.Resize(S.get_S(), XT.Width());
                S.apply(XT, ZT, sketch::columnwise_tag());
                if (_scale_maps)
                    El::Scale<compute_type, compute_type>
                        (sqrt(double(S.get_S()) / _feature_size), ZT);
                base::RowView(VW, _W, starts, S.get_S());
                starts += S.get_S();
                El::Gemm(El::ADJOINT, El::NORMAL, compute_type(1.0), VW, ZT,
                    compute_type(1.0), DV);
            }
            DummyDecode(El::ADJOINT, DV, LP, _rcoding);

        } else {

            El::DistMatrix<compute_type> ZT, VW;
            El::Zeros(DV, XT.Height(), _output_size);
            El::Int starts = 0;
            for(int i = 0; i < _feature_transforms.size(); i++) {
                const sketch_type &S = _feature_transforms[i];
                S.apply(XT, ZT, sketch::rowwise_tag());
                if (_scale_maps)
                    El::Scale<compute_type, compute_type>
                        (sqrt(double(S.get_S()) / _feature_size), ZT);
                base::RowView(VW, _W, starts, S.get_S());
                starts += S.get_S();
                El::Gemm(El::NORMAL, El::NORMAL, compute_type(1.0), ZT, VW,
                    compute_type(1.0), DV);
            }
            DummyDecode(El::NORMAL, DV, LP, _rcoding);

        }
    }

    boost::property_tree::ptree to_ptree() const {
        boost::property_tree::ptree pt;

        pt.put("skylark_object_type", "model:feature_expansion");
        pt.put("skylark_version", VERSION);

        pt.put("input_size", _input_size);
        pt.put("num_outputs", _output_size);
        pt.put("regression", false);

        boost::property_tree::ptree rcoding;
        for(int i = 0; i < _rcoding.size(); i++)
            rcoding.put(std::to_string(i), _rcoding[i]);
        pt.add_child("rcoding", rcoding);

        boost::property_tree::ptree ptfmap;
        ptfmap.put("number_transforms", _feature_transforms.size());
        ptfmap.put("scale_maps", _scale_maps);

        boost::property_tree::ptree ptmaps;
        for(El::Int i = 0; i < _feature_transforms.size(); i++)
            ptmaps.push_back(std::make_pair(std::to_string(i),
                    _feature_transforms[i].to_ptree()));
        ptfmap.add_child("transforms", ptmaps);
        pt.add_child("feature_mapping", ptfmap);

        std::stringstream sW;
        El::Print(_W, "", sW);
        pt.put("weights", sW.str());

        return pt;
    }

    void save(const std::string& fname, const std::string& header) const {
        boost::property_tree::ptree pt = to_ptree();
        std::ofstream of(fname);
        of << header;
        boost::property_tree::write_json(of, pt);
        of.close();
    }

    virtual ~feature_expansion_model_t() {

    }

    El::Int get_input_size() const {
        return _input_size;
    }

    void get_column_coding(std::vector<OutType> &rcoding) const {
        rcoding.resize(_rcoding.size());
        for(int i = 0; i < _rcoding.size(); i++)
            rcoding[i] = _rcoding[i];
    }

protected:
    void build_from_ptree(const boost::property_tree::ptree &pt) {

        _input_size = pt.get<El::Int>("input_size");
        _output_size = pt.get<El::Int>("num_outputs");
        _rcoding.resize(_output_size);
        const boost::property_tree::ptree &ptrcoding =
            pt.get_child("rcoding");
        for(El::Int i = 0; i < _output_size; i++)
            _rcoding[i] = ptrcoding.get<OutType>(std::to_string(i));

        int num_transforms = pt.get<int>("feature_mapping.number_transforms");
        _scale_maps = pt.get<bool>("feature_mapping.scale_maps");

        El::Int s = 0;
        _feature_transforms.resize(num_transforms);
        const boost::property_tree::ptree &ptmaps =
            pt.get_child("feature_mapping.transforms");
        for(int i = 0; i < num_transforms; i++) {
            _feature_transforms[i] =
                sketch_type(sketch_type::from_ptree(ptmaps.get_child(std::to_string(i))));
            s += _feature_transforms[i].get_S();
        }

        std::istringstream W_str(pt.get<std::string>("weights"));
        if (_W.Grid().Size() == 1) {
            _W.Resize(s, _output_size);
            compute_type *buffer = _W.Buffer();
            int ldim = _W.LDim();
            for(int i = 0; i < s; i++) {
                std::string line;
                std::getline(W_str, line);
                std::istringstream Wstream(line);
                for(int j = 0; j < _output_size; j++) {
                    std::string token;
                    Wstream >> token;
                    buffer[i + j * ldim] = atof(token.c_str());
                }
            }
        } else {
            // TODO: can do more memory efficient
            El::DistMatrix<compute_type, El::CIRC, El::CIRC> W0(s, _output_size);
            if (W0.Grid().Rank() == 0) {
                compute_type *buffer = W0.Buffer();
                int ldim = W0.LDim();
                for(int i = 0; i < s; i++) {
                    std::string line;
                    std::getline(W_str, line);
                    std::istringstream Wstream(line);
                    for(int j = 0; j < _output_size; j++) {
                        std::string token;
                        Wstream >> token;
                        buffer[i + j * ldim] = atof(token.c_str());
                    }
                }
            }

            _W = W0;
        }
    }

private:
    El::DistMatrix<compute_type> _W;
    std::vector<OutType> _rcoding;
    bool _scale_maps;
    std::vector<sketch_type> _feature_transforms;
    El::Int _input_size, _output_size;
    El::Int _feature_size;
};

/******************************************************************************/

/**
 * Container -
 * Generic (abstract) model.
 */
template<typename OutType, typename ComputeType = OutType,
         typename dummy = OutType>
struct model_container_t;

/**
 * Container -
 * Generic (abstract) model for continious output - regression
 */
template<typename OutType, typename ComputeType>
struct model_container_t<OutType, ComputeType,
  typename std::enable_if<std::is_floating_point<OutType>::value, OutType>::type> :
public model_t<OutType, ComputeType>
{
    typedef model_t<OutType, ComputeType> model_type;

    model_container_t(const std::shared_ptr<model_type> m) :
        _m(m) {
    }

    model_container_t(const boost::property_tree::ptree &pt) {
        std::string type = pt.get<std::string>("skylark_object_type");

        if (type == "model:kernel")
            _m.reset(new kernel_model_t<skylark::ml::kernel_container_t,
                OutType, ComputeType>(pt));

        if (type == "model:feature_expansion")
            _m.reset(new feature_expansion_model_t<
                sketch::sketch_transform_container_t,
                OutType, ComputeType>(pt));
    }

    virtual void predict(base::direction_t direction_XT,
        const El::DistMatrix<ComputeType> &XT, El::DistMatrix<OutType> &YP) const {
        _m->predict(direction_XT, XT, YP);
    }

    virtual boost::property_tree::ptree to_ptree() const {
        return _m->to_ptree();
    }

    virtual void save(const std::string& fname, const std::string& header)
        const {
        _m->save(fname, header);
    }

    virtual El::Int get_input_size() const {
        return _m->get_input_size();
    }

    virtual ~model_container_t() {

    }

private:
    std::shared_ptr<model_type> _m;
};

/**
 * Container -
 * Generic (abstract) model for discrete (all other) output - classification
 */
template<typename OutType, typename ComputeType>
struct model_container_t<OutType, ComputeType,
  typename std::enable_if<!std::is_floating_point<OutType>::value, OutType>::type> :
public model_t<OutType, ComputeType>
{
    typedef model_t<OutType, ComputeType> model_type;

    model_container_t(const std::shared_ptr<model_type> m) :
        _m(m) {
    }

    model_container_t(const boost::property_tree::ptree &pt) {
        std::string type = pt.get<std::string>("skylark_object_type");

        if (type == "model:kernel")
            _m.reset(new kernel_model_t<skylark::ml::kernel_container_t,
                OutType, ComputeType>(pt));

        if (type == "model:feature_expansion")
            _m.reset(new feature_expansion_model_t<
                sketch::sketch_transform_container_t,
                OutType, ComputeType>(pt));
    }

    virtual void predict(base::direction_t direction_XT,
        const El::DistMatrix<ComputeType> &XT, El::DistMatrix<OutType> &LP,
        El::DistMatrix<ComputeType> &DV) const {
        _m->predict(direction_XT, XT, LP, DV);
    }

    virtual boost::property_tree::ptree to_ptree() const {
        return _m->to_ptree();
    }

    virtual void save(const std::string& fname, const std::string& header)
        const {
        _m->save(fname, header);
    }

    virtual El::Int get_input_size() const {
        return _m->get_input_size();
    }

    virtual void get_column_coding(std::vector<OutType> &rcoding) const {
        return _m->get_column_coding(rcoding);
    }

    virtual ~model_container_t() {

    }

private:
    std::shared_ptr<model_type> _m;
};

} }

#endif /* SKYLARK_ML_MODEL_HPP */
