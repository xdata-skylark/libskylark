#ifndef SKYLARK_BLOCKADMM_HPP
#define SKYLARK_BLOCKADMM_HPP

#include <elemental.hpp>
#include <skylark.hpp>
#include <cmath>
#include <boost/mpi.hpp>

#ifdef SKYLARK_HAVE_OPENMP
#include <omp.h>
#endif

#include "../utility/timer.hpp"

#include "hilbert.hpp"

// Columns are examples, rows are features
typedef elem::DistMatrix<double, elem::STAR, elem::VC> DistInputMatrixType;

// Rows are examples, columns are target values
typedef elem::DistMatrix<double, elem::VC, elem::STAR> DistTargetMatrixType;


typedef elem::DistMatrix<double, elem::VC, elem::STAR> DistMatrixType;
typedef elem::DistMatrix<double, elem::STAR, elem::VC> DistMatrixTypeT;

typedef elem::Matrix<double> LocalMatrixType;


class BlockADMMSolver
{
public:

    typedef skylark::sketch::sketch_transform_t<LocalMatrixType, LocalMatrixType>
    feature_transform_t;
    typedef std::vector<const feature_transform_t *> feature_transform_array_t;


    // No feature transdeforms (aka just linear regression).
    BlockADMMSolver(skylark::sketch::context_t& context,
            const lossfunction* loss,
            const regularization* regularizer,
            double lambda, // regularization parameter
            int NumFeatures,
            int NumFeaturePartitions = 1);

    // Easy interface, aka kernel based.
    template<typename Kernel, typename MapTypeTag>
    BlockADMMSolver(skylark::sketch::context_t& context,
            const lossfunction* loss,
            const regularization* regularizer,
            double lambda, // regularization parameter
            int NumFeatures,
            Kernel kernel,
            MapTypeTag tag,
            int NumFeaturePartitions = 1);

    // Guru interface.
    BlockADMMSolver(skylark::sketch::context_t& context,
            const lossfunction* loss,
            const regularization* regularizer,
            const feature_transform_array_t& featureMaps,
            double lambda, // regularization parameter
            bool ScaleFeatureMaps = true);

    void set_nthreads(int NumThreads) { this->NumThreads = NumThreads; }
    void set_rho(double RHO) { this->RHO = RHO; }
    void set_maxiter(double MAXITER) { this->MAXITER = MAXITER; }
    void set_tol(double TOL) { this->TOL = TOL; }
    void set_cache_transform(bool CacheTransforms) {this->CacheTransforms = CacheTransforms;}

    ~BlockADMMSolver();

    void InitializeFactorizationCache();
    void InitializeTransformCache(int n);

    int train(DistInputMatrixType& X, DistTargetMatrixType& Y, LocalMatrixType& W, DistInputMatrixType& Xv, DistTargetMatrixType& Yv);
    void predict(DistInputMatrixType& X, DistTargetMatrixType& Y,LocalMatrixType& W);
    double evaluate(DistTargetMatrixType& Y, DistTargetMatrixType& Yp);

private:
    skylark::sketch::context_t& context;

    feature_transform_array_t featureMaps;
    int NumFeatures;
    int NumFeaturePartitions;
    lossfunction* loss;
    regularization* regularizer;
    std::vector<int> starts, finishes;
    bool ScaleFeatureMaps;
    bool OwnFeatureMaps;
    LocalMatrixType **Cache;
    LocalMatrixType **TransformCache;
    int NumThreads;

    double lambda;
    double RHO;
    int MAXITER;
    double TOL;

    bool CacheTransforms;
};


void BlockADMMSolver::InitializeFactorizationCache() {
    Cache = new LocalMatrixType* [NumFeaturePartitions];
    for(int j=0; j<NumFeaturePartitions; j++) {
        int start = starts[j];
        int finish = finishes[j];
        int sj = finish - start  + 1;
        Cache[j]  = new elem::Matrix<double>(sj, sj);
    }
}



void BlockADMMSolver::InitializeTransformCache(int n) {
    TransformCache = new LocalMatrixType* [NumFeaturePartitions];
    for(int j=0; j<NumFeaturePartitions; j++) {
        int start = starts[j];
        int finish = finishes[j];
        int sj = finish - start  + 1;
        TransformCache[j]  = new elem::Matrix<double>(sj, n);
    }
}


// No feature transforms (aka just linear regression).
BlockADMMSolver::BlockADMMSolver(skylark::sketch::context_t& context,
        const lossfunction* loss,
        const regularization* regularizer,
        double lambda, // regularization parameter
        int NumFeatures,
        int NumFeaturePartitions) : context(context), NumFeatures(NumFeatures), NumFeaturePartitions(NumFeaturePartitions),
                starts(NumFeaturePartitions), finishes(NumFeaturePartitions),
                NumThreads(1), RHO(1.0), MAXITER(1000), TOL(0.1) {

    this->loss = const_cast<lossfunction *> (loss);
    this->regularizer = const_cast<regularization *> (regularizer);
    this->lambda = lambda;
    this->NumFeaturePartitions = NumFeaturePartitions;
    int blksize = int(ceil(double(NumFeatures) / NumFeaturePartitions));
    for(int i = 0; i < NumFeaturePartitions; i++) {
        starts[i] = i * blksize;
        finishes[i] = std::min((i + 1) * blksize, NumFeatures) - 1;
    }
    this->ScaleFeatureMaps = false;
    OwnFeatureMaps = false;
    InitializeFactorizationCache();
    CacheTransforms = false;
}

// Easy interface, aka kernel based.
template<typename Kernel, typename MapTypeTag>
BlockADMMSolver::BlockADMMSolver(skylark::sketch::context_t& context,
        const lossfunction* loss,
        const regularization* regularizer,
        double lambda, // regularization parameter
        int NumFeatures,
        Kernel kernel,
        MapTypeTag tag,
        int NumFeaturePartitions) : context(context), featureMaps(NumFeaturePartitions),
        NumFeatures(NumFeatures), NumFeaturePartitions(NumFeaturePartitions),
        starts(NumFeaturePartitions), finishes(NumFeaturePartitions),
        NumThreads(1), RHO(1.0), MAXITER(1000), TOL(0.1) {

    this->loss = const_cast<lossfunction *> (loss);
    this->regularizer = const_cast<regularization *> (regularizer);
    this->lambda = lambda;
    int blksize = int(ceil(double(NumFeatures) / NumFeaturePartitions));
    for(int i = 0; i < NumFeaturePartitions; i++) {
        starts[i] = i * blksize;
        finishes[i] = std::min((i + 1) * blksize, NumFeatures) - 1;
        int sj = finishes[i] - starts[i] + 1;
        featureMaps[i] = kernel.template create_rft< LocalMatrixType, LocalMatrixType >(sj, tag, context);
    }
    this->ScaleFeatureMaps = true;
    OwnFeatureMaps = true;
    InitializeFactorizationCache();
    CacheTransforms = false;
}

// Guru interface
BlockADMMSolver::BlockADMMSolver(skylark::sketch::context_t& context,
        const lossfunction* loss,
        const regularization* regularizer,
        const feature_transform_array_t &featureMaps,
        double lambda,
        bool ScaleFeatureMaps) : context(context), featureMaps(featureMaps), NumFeaturePartitions(featureMaps.size()),
                starts(NumFeaturePartitions), finishes(NumFeaturePartitions),
                NumThreads(1), RHO(1.0), MAXITER(1000), TOL(0.1)  {

    this->loss = const_cast<lossfunction *> (loss);
    this->regularizer = const_cast<regularization *> (regularizer);
    this->lambda = lambda;
    NumFeaturePartitions = featureMaps.size();
    NumFeatures = 0;
    for(int i = 0; i < NumFeaturePartitions; i++) {
        starts[i] = NumFeatures;
        finishes[i] = NumFeatures + featureMaps[i]->get_S() - 1;
        NumFeatures += featureMaps[i]->get_S();

        std::cout << starts[i] << " " << finishes[i] << "\n";
    }
    this->ScaleFeatureMaps = ScaleFeatureMaps;
    OwnFeatureMaps = false;
    InitializeFactorizationCache();
    CacheTransforms = false;
}

BlockADMMSolver::~BlockADMMSolver() {
    for(int i=0; i  < NumFeaturePartitions; i++) {
        delete Cache[i];
        if (OwnFeatureMaps)
            delete featureMaps[i];
    }
    delete Cache;
}


int BlockADMMSolver::train(DistInputMatrixType& X, DistTargetMatrixType& Y, LocalMatrixType& Wbar, DistInputMatrixType& Xv, DistTargetMatrixType& Yv) {

    int P = context.size;

    int n = X.Width();
    int d = X.Height();
    int k = Wbar.Width();
    // number of classes, targets - to generalize

    int D = NumFeatures;

    // exception: check if D = Wbar.Height();

    DistMatrixTypeT O(k, n); //uses default Grid
    elem::Zeros(O, k, n);


    DistMatrixTypeT Obar(k, n); //uses default Grid
    elem::Zeros(Obar, k, n);


    DistMatrixTypeT nu(k, n); //uses default Grid
    elem::Zeros(nu, k, n);

    LocalMatrixType W, mu, Wi, mu_ij, ZtObar_ij;

    if(context.rank==0) {
        elem::Zeros(W,  D, k);
        elem::Zeros(mu, D, k);
    }
    elem::Zeros(Wi, D, k);
    elem::Zeros(mu_ij, D, k);
    elem::Zeros(ZtObar_ij, D, k);

    int iter = 0;

    int ni = O.LocalWidth();

    elem::Matrix<double> x = X.Matrix();
    elem::Matrix<double> y = Y.Matrix();


    double localloss = loss->evaluate(O.Matrix(), y);
    double totalloss, accuracy, obj;

    int Dk = D*k;
    int nik  = ni*k;
    int start, finish, sj;

    boost::mpi::timer timer;

    LocalMatrixType sum_o, del_o, wbar_output;
    elem::Zeros(del_o, k, ni);
    DistTargetMatrixType Yp(Yv.Height(), k);
    
    LocalMatrixType wbar_tmp;
    if (NumThreads > 1)
    	elem::Zeros(wbar_tmp, k, ni);
    
    if (CacheTransforms)
                InitializeTransformCache(ni);

    SKYLARK_TIMER_INITIALIZE(ITERATIONS_PROFILE);
    SKYLARK_TIMER_INITIALIZE(COMMUNICATION_PROFILE);
    SKYLARK_TIMER_INITIALIZE(TRANSFORM_PROFILE);
    SKYLARK_TIMER_INITIALIZE(ZTRANSFORM_PROFILE);
    SKYLARK_TIMER_INITIALIZE(ZMULT_PROFILE);
    SKYLARK_TIMER_INITIALIZE(PROXLOSS_PROFILE);
    SKYLARK_TIMER_INITIALIZE(BARRIER_PROFILE);
    SKYLARK_TIMER_INITIALIZE(PREDICTION_PROFILE);

    while(iter<MAXITER) {

        SKYLARK_TIMER_RESTART(ITERATIONS_PROFILE);
		
		iter++;	

        SKYLARK_TIMER_RESTART(COMMUNICATION_PROFILE);
		broadcast(context.comm, Wbar.Buffer(), Dk, 0);
		SKYLARK_TIMER_ACCUMULATE(COMMUNICATION_PROFILE)

        // mu_ij = mu_ij - Wbar
        elem::Axpy(-1.0, Wbar, mu_ij);

        // Obar = Obar - nu
        elem::Axpy(-1.0, nu.Matrix(), Obar.Matrix());

        SKYLARK_TIMER_RESTART(PROXLOSS_PROFILE);
        loss->proxoperator(Obar.Matrix(), 1.0/RHO, y, O.Matrix());
        SKYLARK_TIMER_ACCUMULATE(PROXLOSS_PROFILE);

        if(context.rank==0) {
            regularizer->proxoperator(Wbar, lambda/RHO, mu, W);
        }

        elem::Zeros(sum_o, k, ni);
        elem::Zeros(wbar_output, k, ni);

        int j;
        const feature_transform_t* featureMap;

        SKYLARK_TIMER_RESTART(TRANSFORM_PROFILE);

#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp parallel for if(NumThreads > 1) private(j, start, finish, sj, featureMap) num_threads(NumThreads)
#       endif
        for(j = 0; j < NumFeaturePartitions; j++) {
            start = starts[j];
            finish = finishes[j];
            sj = finish - start  + 1;

            elem::Matrix<double> z(sj, ni);

            if (CacheTransforms && (iter > 1))
            {
                 elem::View(z,  *TransformCache[j], 0, 0, sj, ni);
            }
            else {
                if (featureMaps.size() > 0) {
                    featureMap = featureMaps[j];

                    SKYLARK_TIMER_RESTART(ZTRANSFORM_PROFILE);
                    featureMap->apply(x, z, skylark::sketch::columnwise_tag());
                    SKYLARK_TIMER_ACCUMULATE(ZTRANSFORM_PROFILE)

                    if (ScaleFeatureMaps)
                        elem::Scal(sqrt(double(sj) / d), z);
                } else
                    elem::View(z, x, start, 0, sj, ni);
            }

            elem::Matrix<double> tmp(sj, k);
            elem::Matrix<double> rhs(sj, k);
            elem::Matrix<double> o(k, ni);

            if(iter==1) {

                elem::Matrix<double> Ones;
                elem::Ones(Ones, sj, 1);
                elem::Gemm(elem::NORMAL, elem::TRANSPOSE, 1.0, z, z, 0.0, *Cache[j]);
                Cache[j]->UpdateDiagonal(Ones);
                elem::Inverse(*Cache[j]);

                if (CacheTransforms) 
                    *TransformCache[j] = z;
            }

            elem::View(tmp, Wbar, start, 0, sj, k); //tmp = Wbar[J,:]
           
            if (NumThreads > 1) {
            	elem::Gemm(elem::TRANSPOSE, elem::NORMAL, 1.0, tmp, z, 0.0, wbar_tmp);
            	
#           	ifdef SKYLARK_HAVE_OPENMP
#           	pragma omp critical
#           	endif
            	elem::Axpy(1.0, wbar_tmp, wbar_output);
            } else 
            	elem::Gemm(elem::TRANSPOSE, elem::NORMAL, 1.0, tmp, z, 1.0, wbar_output);
            
            rhs = tmp; //rhs = Wbar[J,:]
            elem::View(tmp, mu_ij, start, 0, sj, k); //tmp = mu_ij[J,:]
            elem::Axpy(-1.0, tmp, rhs); // rhs = rhs - mu_ij[J,:] = Wbar[J,:] - mu_ij[J,:]
            elem::View(tmp, ZtObar_ij, start, 0, sj, k);
            elem::Axpy(+1.0, tmp, rhs); // rhs = rhs + ZtObar_ij[J,:]

            SKYLARK_TIMER_RESTART(ZMULT_PROFILE);
            elem::Matrix<double> dsum = del_o;
            elem::Axpy(NumFeaturePartitions + 1.0, nu.Matrix(), dsum);
            elem::Gemm(elem::NORMAL, elem::TRANSPOSE, 1.0/(NumFeaturePartitions + 1.0), z, dsum, 1.0, rhs); // rhs = rhs + z'*(1/(n+1) * del_o + nu)
            SKYLARK_TIMER_ACCUMULATE(ZMULT_PROFILE);

            elem::View(tmp, Wi, start, 0, sj, k);
            elem::Gemm(elem::NORMAL, elem::NORMAL, 1.0, *Cache[j], rhs, 0.0, tmp); // ]tmp = Wi[J,:] = Cache[j]*rhs

            SKYLARK_TIMER_RESTART(ZMULT_PROFILE);
            elem::Gemm(elem::TRANSPOSE, elem::NORMAL, 1.0, tmp, z, 0.0, o); // o = (z*tmp)' = (z*Wi[J,:])'
            SKYLARK_TIMER_ACCUMULATE(ZMULT_PROFILE);
            
            // mu_ij[JJ,:] = mu_ij[JJ,:] + Wi[JJ,:];
            elem::View(tmp, mu_ij, start, 0, sj, k); //tmp = mu_ij[J,:]
            elem::View(rhs, Wi, start, 0, sj, k);
            elem::Axpy(+1.0, rhs, tmp);

            //ZtObar_ij[JJ,:] = numpy.dot(Z.T, o);
            elem::View(tmp, ZtObar_ij, start, 0, sj, k);
            elem::Gemm(elem::NORMAL, elem::TRANSPOSE, 1.0, z, o, 0.0, tmp);

            //  sum_o += o
            if (NumThreads > 1) {
#           	ifdef SKYLARK_HAVE_OPENMP
#          		pragma omp critical
#           	endif
            	elem::Axpy(1.0, o, sum_o);
            } else
            	elem::Axpy(1.0, o, sum_o);

            z.Empty();
        }
        
        SKYLARK_TIMER_ACCUMULATE(TRANSFORM_PROFILE);

        localloss = 0.0 ;
        //	elem::Zeros(o, ni, k);
        elem::Matrix<double> o(k, ni);
        elem::MakeZeros(o);
        elem::Scal(-1.0, sum_o);
        elem::Axpy(+1.0, O.Matrix(), sum_o); // sum_o = O.Matrix - sum_o
        del_o = sum_o;

        SKYLARK_TIMER_RESTART(PREDICTION_PROFILE);
        if (Xv.Width() > 0) {
            elem::MakeZeros(Yp);
            predict(Xv, Yp, Wbar);
            accuracy = evaluate(Yv, Yp);
        }
        SKYLARK_TIMER_ACCUMULATE(PREDICTION_PROFILE);

        localloss += loss->evaluate(wbar_output, y);

        SKYLARK_TIMER_RESTART(COMMUNICATION_PROFILE);
        reduce(context.comm, localloss, totalloss, std::plus<double>(), 0);
        SKYLARK_TIMER_ACCUMULATE(COMMUNICATION_PROFILE);

        if(context.rank==0) {
            obj = totalloss + lambda*regularizer->evaluate(Wbar);
            if (Xv.Width()==0) {
                std::cout << "iteration " << iter << " objective " << obj << " time " << timer.elapsed() << " seconds" << std::endl;
            }
            else {
                std::cout << "iteration " << iter << " objective " << obj << " accuracy " << accuracy << " time " << timer.elapsed() << " seconds" << std::endl;
            }
        }

		elem::Copy(O.Matrix(), Obar.Matrix());
		elem::Scal(1.0/(NumFeaturePartitions+1.0), sum_o);
		elem::Axpy(-1.0, sum_o, Obar.Matrix());

		elem::Axpy(+1.0, O.Matrix(), nu.Matrix());
		elem::Axpy(-1.0, Obar.Matrix(), nu.Matrix());

		//Wbar = comm.reduce(Wi)
        SKYLARK_TIMER_RESTART(COMMUNICATION_PROFILE);
		boost::mpi::reduce (context.comm,
		                        Wi.LockedBuffer(),
		                        Wi.MemorySize(),
		                        Wbar.Buffer(),
		                        std::plus<double>(),
		                        0);
        SKYLARK_TIMER_ACCUMULATE(COMMUNICATION_PROFILE);

		if(context.rank==0) {
			//Wbar = (Wisum + W)/(P+1)
			elem::Axpy(1.0, W, Wbar);
			elem::Scal(1.0/(P+1), Wbar);

			// mu = mu + W - Wbar;
			elem::Axpy(+1.0, W, mu);
			elem::Axpy(-1.0, Wbar, mu);
		}

        SKYLARK_TIMER_RESTART(BARRIER_PROFILE);
        context.comm.barrier();
        SKYLARK_TIMER_ACCUMULATE(BARRIER_PROFILE);

        SKYLARK_TIMER_ACCUMULATE(ITERATIONS_PROFILE);
	}

    SKYLARK_TIMER_PRINT(ITERATIONS_PROFILE);
    SKYLARK_TIMER_PRINT(COMMUNICATION_PROFILE);
    SKYLARK_TIMER_PRINT(TRANSFORM_PROFILE);	
    SKYLARK_TIMER_PRINT(ZTRANSFORM_PROFILE);
    SKYLARK_TIMER_PRINT(ZMULT_PROFILE);
    SKYLARK_TIMER_PRINT(PROXLOSS_PROFILE);
    SKYLARK_TIMER_PRINT(BARRIER_PROFILE);
    SKYLARK_TIMER_PRINT(PREDICTION_PROFILE);

	return 0;
}

void BlockADMMSolver::predict(DistInputMatrixType& X, DistTargetMatrixType& Y, LocalMatrixType& W) {

	// TOD W should be really kept as part of the model

	int n = X.Width();
	int d = X.Height();
	int k = Y.Width();
	int ni = X.LocalWidth();
	int j, start, finish, sj;
	const feature_transform_t* featureMap;

	if (featureMaps.size() == 0) {
		Y.Resize(n, k);
		elem::Gemm(elem::TRANSPOSE,elem::NORMAL,1.0, X.Matrix(), W, 0.0, Y.Matrix());
		return;
	}

	elem::Zeros(Y, n, k);

	LocalMatrixType Wslice;


#   ifdef SKYLARK_HAVE_OPENMP
#   pragma omp parallel for if(NumThreads > 1) private(j, start, finish, sj, featureMap) num_threads(NumThreads)
#   endif
	for(j = 0; j < NumFeaturePartitions; j++) {
		start = starts[j];
		finish = finishes[j];
		sj = finish - start  + 1;

		elem::Matrix<double> z(sj, ni);
		featureMap = featureMaps[j];
		featureMap->apply(X.Matrix(), z, skylark::sketch::columnwise_tag());
		if (ScaleFeatureMaps)
			elem::Scal(sqrt(double(sj) / d), z);

		elem::Matrix<double> o(ni, k);

		elem::View(Wslice, W, start, 0, sj, k);
		elem::Gemm(elem::TRANSPOSE, elem::NORMAL, 1.0, z, Wslice, 0.0, o);
		
#       ifdef SKYLARK_HAVE_OPENMP
#       pragma omp critical
#       endif
		elem::Axpy(+1.0, o, Y.Matrix());
	}
}


double BlockADMMSolver::evaluate(DistTargetMatrixType& Yt, DistTargetMatrixType& Yp) {
    int correct = 0;
             double o, o1;
             int pred;
             double accuracy = 0.0;

             for(int i=0; i < Yp.LocalHeight(); i++) {
                 o = Yp.GetLocal(i,0);
                 pred = 0;
                 for(int j=1; j < Yp.Width(); j++) {
                     o1 = Yp.GetLocal(i,j);
                     if ( o1 > o) {
                         o = o1;
                         pred = j;
                     }
                 }

                 if(pred == (int) Yt.GetLocal(i,0))
                     correct++;
              }

             int totalcorrect;
             boost::mpi::reduce(context.comm, correct, totalcorrect, std::plus<double>(), 0);
             if(context.rank ==0)
                 accuracy =  totalcorrect*100.0/Yt.Height();

             return accuracy;
}

#endif /* SKYLARK_BLOCKADDM_HPP */
