/*
 * run.hpp
 *
 *  Created on: Mar 17, 2014
 *      Author: vikas
 */

#ifndef RUN_HPP_
#define RUN_HPP_
#include "BlockADMM.hpp"
#include "options.hpp"
#include "io.hpp"

BlockADMMSolver* GetSolver(skylark::sketch::context_t& context, hilbert_options_t& options, int dimensions) {

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
       default:
           // TODO
           break;
       }

        BlockADMMSolver *Solver = NULL;
        int features = 0;
        switch(options.kernel) {
        case LINEAR:
            features = dimensions;
            Solver = new BlockADMMSolver(
                    context,
                    loss,
                    regularizer,
                    options.lambda,
                    dimensions,
                    options.numfeaturepartitions);
            break;

        case GAUSSIAN:
            features = options.randomfeatures;
            if (options.regularmap)
                Solver = new BlockADMMSolver(
                        context,
                        loss,
                        regularizer,
                        options.lambda,
                        features,
                        skylark::ml::kernels::gaussian_t(dimensions, options.kernelparam),
                        skylark::ml::regular_feature_transform_tag(),
                        options.numfeaturepartitions);

            else
                Solver = new BlockADMMSolver(
                        context,
                        loss    ,
                        regularizer,
                        options.lambda,
                        features,
                        skylark::ml::kernels::gaussian_t(dimensions, options.kernelparam),
                        skylark::ml::fast_feature_transform_tag(),
                        options.numfeaturepartitions);
            break;

        case POLYNOMIAL:
            features = options.randomfeatures;
            Solver = new BlockADMMSolver(
                    context,
                    loss,
                    regularizer,
                    options.lambda,
                    features,
                    skylark::ml::kernels::polynomial_t(dimensions, options.kernelparam, options.kernelparam2, options.kernelparam3),
                    skylark::ml::regular_feature_transform_tag(),
                    options.numfeaturepartitions);
            break;

        case LAPLACIAN:
            features = options.randomfeatures;
            Solver = new BlockADMMSolver(
                    context,
                    loss,
                    regularizer,
                    options.lambda,
                    features,
                    skylark::ml::kernels::laplacian_t(dimensions, options.kernelparam),
                    skylark::ml::regular_feature_transform_tag(),
                    options.numfeaturepartitions);
            break;

        case EXPSEMIGROUP:
            features = options.randomfeatures;
            Solver = new BlockADMMSolver(
                    context,
                    loss,
                    regularizer,
                    options.lambda,
                    features,
                    skylark::ml::kernels::expsemigroup_t(dimensions, options.kernelparam),
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

        return Solver;
}


int max(DistTargetMatrixType& Y) {
    int k =  *std::max_element(Y.Buffer(), Y.Buffer() + Y.LocalHeight());
    return k;
}
int max(elem::Matrix<double> Y) {
    int k =  *std::max_element(Y.Buffer(), Y.Buffer() + Y.Height());
}

template<class LabelType>
int GetNumClasses(skylark::sketch::context_t& context, LabelType& Y) {
    int k;
    int kmax = max(Y);
    boost::mpi::all_reduce(context.comm, kmax, k, boost::mpi::maximum<int>());
    if (k>1) // we assume 0-to-N encoding of classes. Hence N = k+1. For two classes, k=1.
       k++;
    return k;
}

template <class InputType, class LabelType>
int run(skylark::sketch::context_t& context, hilbert_options_t& options) {
    InputType X, Xv, Xt;
    LabelType Y, Yv, Yt;

    read(context, options.fileformat, options.trainfile, X, Y);
    int dimensions = X.Height();
    int classes = GetNumClasses<LabelType>(context,Y);

    BlockADMMSolver* Solver = GetSolver(context, options, dimensions);

    if(!options.valfile.empty()) {
        context.comm.barrier();
        if(context.rank == 0) std::cout << "Loading validation data." << std::endl;
        read(context, options.fileformat, options.valfile, Xv, Yv);
    }

    std::cout << "Dimensions =" << dimensions << " Classes = " << classes  << std::endl;

    elem::Matrix<double> Wbar(dimensions, classes);
    elem::MakeZeros(Wbar);

    Solver->train(X, Y, Wbar, Xv, Yv);

    SaveModel(context, options, Wbar);

    if(!options.testfile.empty()) {
        context.comm.barrier();
        if(context.rank == 0) std::cout << "Starting testing phase." << std::endl;
        read(context, options.fileformat, options.testfile, Xt, Yt);

        LabelType Yp(Yt.Height(), classes);
        Solver->predict(Xt, Yp, Wbar);
        double accuracy = Solver->evaluate(Yt, Yp);
        if(context.rank == 0) std::cout << "Test Accuracy = " <<  accuracy << " %" << std::endl;
    }

    return 0;
}



#endif /* RUN_HPP_ */
