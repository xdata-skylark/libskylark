#ifndef SKYLARK_HILBERT_HPP
#define SKYLARK_HILBERT_HPP

#include "FunctionProx.hpp"
#include "utils.hpp"
#include "model.hpp"
#include "io.hpp"
#include "BlockADMM.hpp"
#include "options.hpp"

#include "../base/context.hpp"
#include "model.hpp"


template <class InputType>
BlockADMMSolver<InputType>* GetSolver(skylark::base::context_t& context,
    const hilbert_options_t& options, int dimensions) {

    lossfunction *loss = NULL;
    switch(options.lossfunction) {
    case SQUARED:
        loss = new squaredloss();
        break;
    case HINGE:
        loss = new hingeloss();
        break;
    case LOGISTIC:
        loss = new logisticloss();
        break;
    case LAD:
        loss = new ladloss();
        break;
    default:
        // TODO
        break;
    }

    regularization *regularizer = NULL;
    switch(options.regularizer) {
    case L2:
        regularizer = new l2();
        break;
    case L1:
    	regularizer = new l1();
    	break;
    default:
        // TODO
        break;
    }

    BlockADMMSolver<InputType> *Solver = NULL;
    int features = 0;
    switch(options.kernel) {
    case LINEAR:
        features = dimensions;
        Solver =
            new BlockADMMSolver<InputType>(loss,
                regularizer,
                options.lambda,
                dimensions,
                options.numfeaturepartitions);
        break;

    case GAUSSIAN:
        features = options.randomfeatures;
        if (!options.usefast)
            if (options.seqtype == LEAPED_HALTON)
                Solver =
                    new BlockADMMSolver<InputType>(context,
                        loss,
                        regularizer,
                        options.lambda,
                        features,
                        skylark::ml::gaussian_t(dimensions,
                            options.kernelparam),
                        skylark::ml::quasi_feature_transform_tag(),
                        options.numfeaturepartitions);
            else
                Solver =
                    new BlockADMMSolver<InputType>(context,
                        loss,
                        regularizer,
                        options.lambda,
                        features,
                        skylark::ml::gaussian_t(dimensions,
                            options.kernelparam),
                        skylark::ml::regular_feature_transform_tag(),
                        options.numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    regularizer,
                    options.lambda,
                    features,
                    skylark::ml::gaussian_t(dimensions,
                        options.kernelparam),
                    skylark::ml::fast_feature_transform_tag(),
                    options.numfeaturepartitions);
        break;

    case POLYNOMIAL:
        features = options.randomfeatures;
        Solver = 
            new BlockADMMSolver<InputType>(context,
                loss,
                regularizer,
                options.lambda,
                features,
                skylark::ml::polynomial_t(dimensions,
                    options.kernelparam, options.kernelparam2, options.kernelparam3),
                skylark::ml::regular_feature_transform_tag(),
                options.numfeaturepartitions);
        break;

    case MATERN:
        features = options.randomfeatures;
        if (!options.usefast)
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    regularizer,
                    options.lambda,
                    features,
                    skylark::ml::matern_t(dimensions,
                        options.kernelparam, options.kernelparam2),
                    skylark::ml::regular_feature_transform_tag(),
                    options.numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    regularizer,
                    options.lambda,
                    features,
                    skylark::ml::matern_t(dimensions,
                        options.kernelparam, options.kernelparam2),
                    skylark::ml::fast_feature_transform_tag(),
                    options.numfeaturepartitions);
        break;

    case LAPLACIAN:
        features = options.randomfeatures;
        if (options.seqtype == LEAPED_HALTON)
            new BlockADMMSolver<InputType>(context,
                loss,
                regularizer,
                options.lambda,
                features,
                skylark::ml::laplacian_t(dimensions,
                    options.kernelparam),
                skylark::ml::quasi_feature_transform_tag(),
                options.numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    regularizer,
                    options.lambda,
                    features,
                    skylark::ml::laplacian_t(dimensions,
                        options.kernelparam),
                    skylark::ml::regular_feature_transform_tag(),
                    options.numfeaturepartitions);

        break;

    case EXPSEMIGROUP:
        features = options.randomfeatures;
        if (options.seqtype == LEAPED_HALTON)
            new BlockADMMSolver<InputType>(context,
                loss,
                regularizer,
                options.lambda,
                features,
                skylark::ml::expsemigroup_t(dimensions,
                    options.kernelparam),
                skylark::ml::quasi_feature_transform_tag(),
                options.numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    regularizer,
                    options.lambda,
                    features,
                    skylark::ml::expsemigroup_t(dimensions,
                        options.kernelparam),
                    skylark::ml::regular_feature_transform_tag(),
                    options.numfeaturepartitions);
        break;

    default:
        // TODO!
        break;

    }

    // Set parameters
    Solver->set_rho(options.rho);
    Solver->set_maxiter(options.MAXITER);
    Solver->set_tol(options.tolerance);
    Solver->set_nthreads(options.numthreads);
    Solver->set_cache_transform(options.cachetransforms);

    return Solver;
}


void ShiftForLogistic(LocalMatrixType& Y) {
    double y;
    for(int i=0;i<Y.Height(); i++) {
        y = Y.Get(i, 0);
        Y.Set(i, 0, 0.5*(y+1.0));
    }
}

namespace skylark { namespace ml {

template <class InputType, class LabelType>
void LargeScaleKernelLearning(const boost::mpi::communicator& comm,
    InputType &X, LabelType &Y, skylark::base::context_t& context,
    hilbert_options_t& options) {

    int rank = comm.rank();

    InputType Xv;
    LabelType Yv;


    int dimensions = skylark::base::Height(X);
    int targets = options.regression ? 1 : GetNumTargets<LabelType>(comm, Y);
    bool shift = false;

    if (!options.regression && options.lossfunction == LOGISTIC
        && targets == 1) {
        ShiftForLogistic(Y);
        targets = 2;
        shift = true;
    }

    BlockADMMSolver<InputType>* Solver =
        GetSolver<InputType>(context, options, dimensions);

    if(!options.valfile.empty()) {
        comm.barrier();
        if(rank == 0)
            std::cout << "Loading validation data." << std::endl;

        read(comm, options.fileformat, options.valfile, Xv, Yv,
            skylark::base::Height(X));

        if ((options.lossfunction == LOGISTIC) && shift) 
            ShiftForLogistic(Yv);
    }

    skylark::ml::model_t* model =
        Solver->train(X, Y, Xv, Yv, options.regression, comm);

    // TODO should be done "outside"
    if (comm.rank() == 0)
        model->save(options.modelfile, options.print());
}

} }

#endif /* SKYLARK_HILBERT_DRIVER_HPP */
