#ifndef SKYLARK_HILBERT_HPP
#define SKYLARK_HILBERT_HPP

#include "utils.hpp"
#include "model.hpp"
#include "io.hpp"
#include "BlockADMM.hpp"
#include "options.hpp"

template <class InputType>
BlockADMMSolver<InputType>* GetSolver(skylark::base::context_t& context,
    const hilbert_options_t& options, int dimensions) {

    typedef typename BlockADMMSolver<InputType>::value_type value_type;

    skylark::algorithms::loss_t<value_type> *loss = NULL;
    switch(options.lossfunction) {
    case SQUARED:
        loss = new skylark::algorithms::squared_loss_t<value_type>();
        break;
    case HINGE:
        loss = new skylark::algorithms::hinge_loss_t<value_type>();
        break;
    case LOGISTIC:
        loss = new skylark::algorithms::logistic_loss_t<value_type>();
        break;
    case LAD:
        loss = new skylark::algorithms::lad_loss_t<value_type>();
        break;
    default:
        // TODO
        break;
    }

    skylark::algorithms::regularizer_t<value_type> *regularizer = NULL;
    if (options.lambda == 0 || options.regularizer == NOREG)
        regularizer = new skylark::algorithms::empty_regularizer_t<value_type>();
    else
        switch(options.regularizer) {
        case L2:
            regularizer = new skylark::algorithms::l2_regularizer_t<value_type>();
            break;
        case L1:
            regularizer = new skylark::algorithms::l1_regularizer_t<value_type>();
            break;
        default:
            // TODO
            break;
        }

    BlockADMMSolver<InputType> *Solver = NULL;
    int features = 0;
    switch(options.kernel) {
    case LINEAR:
        features =
            (options.randomfeatures == 0 ? dimensions : options.randomfeatures);
        if (options.randomfeatures == 0)
            Solver =
                new BlockADMMSolver<InputType>(loss,
                    regularizer,
                    options.lambda,
                    dimensions,
                    options.numfeaturepartitions);
        else
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    regularizer,
                    options.lambda,
                    features,
                    skylark::ml::linear_t(dimensions),
                    skylark::ml::sparse_feature_transform_tag(),
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


void ShiftForLogistic(El::Matrix<double>& Y) {
    double y;
    for(int i=0;i<Y.Height(); i++) {
        y = Y.Get(i, 0);
        Y.Set(i, 0, 0.5*(y+1.0));
    }
}

namespace skylark { namespace ml {

template <class InputType, class LabelType>
void LargeScaleKernelLearning(const boost::mpi::communicator& comm,
    InputType &X, LabelType &Y, InputType &Xv, LabelType &Yv,
    skylark::base::context_t& context, const hilbert_options_t& options) {

    // TODO variable number of targets for regression
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

    if (!options.valfile.empty() && (options.lossfunction == LOGISTIC) && shift) 
        ShiftForLogistic(Yv);

    skylark::ml::hilbert_model_t* model =
        Solver->train(X, Y, Xv, Yv, options.regression, comm);

    // TODO unshift logistic

    // TODO should be done "outside"
    if (comm.rank() == 0)
        model->save(options.modelfile, options.print());
}

} }

#endif /* SKYLARK_HILBERT_DRIVER_HPP */
