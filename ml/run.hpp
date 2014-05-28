#ifndef SKYLARK_HILBERT_RUN_HPP
#define SKYLARK_HILBERT_RUN_HPP

//#include "hilbert.hpp"
#include "BlockADMM.hpp"
#include "options.hpp"
#include "io.hpp"
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
        if (options.regularmap)
            Solver =
                new BlockADMMSolver<InputType>(context,
                    loss,
                    regularizer,
                    options.lambda,
                    features,
                    skylark::ml::kernels::gaussian_t(dimensions,
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
                    skylark::ml::kernels::gaussian_t(dimensions,
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
                skylark::ml::kernels::polynomial_t(dimensions,
                    options.kernelparam, options.kernelparam2, options.kernelparam3),
                skylark::ml::regular_feature_transform_tag(),
                options.numfeaturepartitions);
        break;

    case LAPLACIAN:
        features = options.randomfeatures;
        Solver = 
            new BlockADMMSolver<InputType>(context,
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
        Solver =
            new BlockADMMSolver<InputType>(context,
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

template <class InputType, class LabelType>
int run(const boost::mpi::communicator& comm, skylark::base::context_t& context,
    hilbert_options_t& options) {

    int rank = comm.rank();

    InputType X, Xv, Xt;
    LabelType Y, Yv, Yt;

    if(!options.trainfile.empty()) { //training mode

    	read(comm, options.fileformat, options.trainfile, X, Y);
    	int dimensions = skylark::base::Height(X);
    	int targets = GetNumTargets<LabelType>(comm, Y);
    	bool shift = false;

    	if ((options.lossfunction == LOGISTIC) && (targets == 1)) {
    		ShiftForLogistic(Y);
    		targets = 2;
    		shift = true;
    	}

    	BlockADMMSolver<InputType>* Solver =
        	GetSolver<InputType>(context, options, dimensions);

    	if(!options.valfile.empty()) {
    		comm.barrier();
    		if(rank == 0) std::cout << "Loading validation data." << std::endl;

    		read(comm, options.fileformat, options.valfile, Xv, Yv,
    			skylark::base::Height(X));

    		if ((options.lossfunction == LOGISTIC) && shift) {
    				ShiftForLogistic(Yv);
            	}
    		}

    	skylark::ml::Model<InputType>* model = Solver->train(X, Y, Xv, Yv, comm);
    	model->save(options.modelfile, options.print(), comm.rank());
    }

    else {

    	std::cout << "Testing Mode" << std::endl;
    	skylark::ml::Model<InputType>* model = new skylark::ml::Model<InputType>(options.modelfile, comm);
    	model->set_num_threads(options.numthreads);
    	read(comm, options.fileformat, options.testfile, Xt, Yt,
    	    				skylark::base::Height(X));

    	int targets = GetNumTargets<LabelType>(comm, Yt);
    	bool shift = false;
    //	if ((model.lossfunction == LOGISTIC) && (targets == 1)) {
    //	    		ShiftForLogistic(Yt);
    //	    		targets = 2;
    //	    		shift = true;
    //	 }

    	LabelType DecisionValues(Yt.Height(), model->get_classes());
    	LabelType PredictedLabels(Yt.Height(), 1);
    	elem::MakeZeros(DecisionValues);
    	elem::MakeZeros(PredictedLabels);

    	std::cout << "Starting predictions" << std::endl;
    	model->predict(Xt, PredictedLabels, DecisionValues);
    	double accuracy = model->evaluate(Yt, DecisionValues, comm);
    	if(rank == 0)
    	        std::cout << "Test Accuracy = " <<  accuracy << " %" << std::endl;

    	// fix logistic case -- provide mechanism to dump predictions -- clean up evaluate

    }
    /*else  { // test mode

    	// set other options from model file

    	options_str =  read_header(comm, options.modelfile);
    	int argc1;
    	char **argv1;

    	argc1 = splitstr(options_str, argv1);
    	hilbert_options_t train_options (argc1, argv1, comm.rank());


    	// read model Wbar
    	elem::Matrix<double> Wbar;
    	elem::Read(Wbar, options.modelfile, elem::ASCII);

    	// Create a solver object
    	int dimensions = Wbar.Height();
    	BlockADMMSolver<InputType>* Solver =
    	        	GetSolver<InputType>(context, options, dimensions);

    	if(!options.testfile.empty()) {
    		comm.barrier();
    		if(rank == 0) std::cout << "Starting testing phase (currently we load all data in memory - to be changed.)" << std::endl;
    		read(comm, options.fileformat, options.testfile, Xt, Yt,
    				skylark::base::Height(X));
        		if ((options.lossfunction == LOGISTIC) && shift) {
        			for(int i=0;i<Yt.Height(); i++) {
        				y = Yt.Get(i, 0);
        				Yt.Set(i, 0, 0.5*(y+1.0));
        					}
                    	}

        		LabelType Yp(Yt.Height(), classes);
        		Solver->predict(Xt, Yp, Wbar);
        		double accuracy = Solver->evaluate(Yt, Yp, comm);
        		if(rank == 0)
        			std::cout << "Test Accuracy = " <<  accuracy << " %" << std::endl;
    	}

    	else {
    		std::cout << "Test mode: did not detect a test file." << std::endl;
    	}
    }
*/


    return 0;
}



#endif /* SKYLARK_HILBERT_RUN_HPP */
