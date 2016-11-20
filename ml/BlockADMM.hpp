#ifndef SKYLARK_BLOCKADMM_HPP
#define SKYLARK_BLOCKADMM_HPP

#include <El.hpp>
#include <skylark.hpp>
#include <cmath>
#include <boost/mpi.hpp>

#ifdef SKYLARK_HAVE_OPENMP
#include <omp.h>
#endif

#include "../utility/timer.hpp"

template <class InputType>
struct BlockADMMSolver {

    typedef typename skylark::utility::typer_t<InputType>::value_type value_type;

    typedef InputType data_matrix_t;
    typedef El::Matrix<value_type> feature_matrix_t;
    typedef El::Matrix<value_type> target_matrix_t;

    typedef skylark::sketch::sketch_transform_t<data_matrix_t, feature_matrix_t>
    feature_transform_t;
    typedef std::vector<const feature_transform_t *> feature_transform_array_t;


    // No feature transdeforms (aka just linear regression).
    BlockADMMSolver(const skylark::algorithms::loss_t<value_type>* loss,
        const skylark::algorithms::regularizer_t<value_type>* regularizer,
        double lambda, // regularization parameter
        int NumFeatures,
        int NumFeaturePartitions = 1);

    // Easy interface, aka kernel based.
    template<typename Kernel, typename MapTypeTag>
    BlockADMMSolver<InputType>(skylark::base::context_t& context,
        const skylark::algorithms::loss_t<value_type>* loss,
        const skylark::algorithms::regularizer_t<value_type>* regularizer,
        double lambda, // regularization parameter
        int NumFeatures,
        Kernel kernel,
        MapTypeTag tag,
        int NumFeaturePartitions = 1);

    // Easy interface, aka kernel based, with quasi-random features.
    template<typename Kernel>
    BlockADMMSolver<InputType>(skylark::base::context_t& context,
        const skylark::algorithms::loss_t<value_type>* loss,
        const skylark::algorithms::regularizer_t<value_type>* regularizer,
        double lambda, // regularization parameter
        int NumFeatures,
        Kernel kernel,
        skylark::ml::quasi_feature_transform_tag tag,
        int NumFeaturePartitions);

    // Guru interface.
    BlockADMMSolver<InputType>(const skylark::algorithms::loss_t<value_type>* loss,
        const skylark::algorithms::regularizer_t<value_type>* regularizer,
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

    skylark::ml::hilbert_model_t* train(data_matrix_t& X,
        target_matrix_t& Y, data_matrix_t& Xv, target_matrix_t& Yv,
        bool regression, const boost::mpi::communicator& comm);

    int get_numfeatures() {return NumFeatures;}

    feature_transform_array_t& get_feature_maps() {return featureMaps;}

private:

    typedef El::Matrix<value_type> local_matrix_t;

    feature_transform_array_t featureMaps;
    int NumFeatures;
    int NumFeaturePartitions;
    const skylark::algorithms::loss_t<value_type>* loss;
    const skylark::algorithms::regularizer_t<value_type>* regularizer;
    std::vector<int> starts, finishes;
    bool ScaleFeatureMaps;
    bool OwnFeatureMaps;
    local_matrix_t **Cache;
    local_matrix_t **TransformCache;
    int NumThreads;

    double lambda;
    double RHO;
    int MAXITER;
    double TOL;

    bool CacheTransforms;
};

template <class InputType>
void BlockADMMSolver<InputType>::InitializeFactorizationCache() {
    Cache = new local_matrix_t*[NumFeaturePartitions];
    for(int j=0; j<NumFeaturePartitions; j++) {
        int start = starts[j];
        int finish = finishes[j];
        int sj = finish - start  + 1;
        Cache[j]  = new local_matrix_t(sj, sj);
    }
}

template <class InputType>
void BlockADMMSolver<InputType>::InitializeTransformCache(int n) {
    TransformCache = new local_matrix_t*[NumFeaturePartitions];
    for(int j=0; j<NumFeaturePartitions; j++) {
        int start = starts[j];
        int finish = finishes[j];
        int sj = finish - start  + 1;
        TransformCache[j]  = new local_matrix_t(sj, n);
    }
}


// No feature transforms (aka just linear regression).
template <class InputType>
BlockADMMSolver<InputType>::BlockADMMSolver(
        const skylark::algorithms::loss_t<value_type>* loss,
        const skylark::algorithms::regularizer_t<value_type>* regularizer,
        double lambda, // regularization parameter
        int NumFeatures,
        int NumFeaturePartitions) :
        NumFeatures(NumFeatures),
        NumFeaturePartitions(NumFeaturePartitions),
        loss(loss), regularizer(regularizer),
        starts(NumFeaturePartitions), finishes(NumFeaturePartitions),
        NumThreads(1), lambda(lambda), RHO(1.0), MAXITER(1000), TOL(0.1) {

    int cstart = 0, nf = NumFeatures, np = NumFeaturePartitions;
    for(int i = 0; i < NumFeaturePartitions; i++) {
        int sj = int(floor(double(nf) / np));
        starts[i] = cstart;
        finishes[i] = cstart + sj - 1;
        cstart += sj;
        nf -= sj;
        np--;
    }
    this->ScaleFeatureMaps = false;
    OwnFeatureMaps = false;
    InitializeFactorizationCache();
    CacheTransforms = false;
}

// Easy interface, aka kernel based.
template<class InputType>
template<typename Kernel, typename MapTypeTag>
BlockADMMSolver<InputType>::BlockADMMSolver(skylark::base::context_t& context,
    const skylark::algorithms::loss_t<value_type>* loss,
    const skylark::algorithms::regularizer_t<value_type>* regularizer,
    double lambda, // regularization parameter
    int NumFeatures,
    Kernel kernel,
    MapTypeTag tag,
    int NumFeaturePartitions) :
    featureMaps(NumFeaturePartitions),
    NumFeatures(NumFeatures), NumFeaturePartitions(NumFeaturePartitions),
    loss(loss), regularizer(regularizer),
    starts(NumFeaturePartitions), finishes(NumFeaturePartitions),
    NumThreads(1), lambda(lambda), RHO(1.0), MAXITER(1000), TOL(0.1) {

    int cstart = 0, nf = NumFeatures, np = NumFeaturePartitions;
    for(int i = 0; i < NumFeaturePartitions; i++) {
        int sj = int(floor(double(nf) / np));
        starts[i] = cstart;
        finishes[i] = cstart + sj - 1;
        cstart += sj;
        nf -= sj;
        np--;

        featureMaps[i] =
            kernel.template create_rft<data_matrix_t, local_matrix_t>(sj, tag, context);
    }
    this->ScaleFeatureMaps = true;
    OwnFeatureMaps = true;
    InitializeFactorizationCache();
    CacheTransforms = false;
}

// Easy interface, aka kernel based, with quasi-random features.
template<class InputType>
template<typename Kernel>
BlockADMMSolver<InputType>::BlockADMMSolver(skylark::base::context_t& context,
    const skylark::algorithms::loss_t<value_type>* loss,
    const skylark::algorithms::regularizer_t<value_type>* regularizer,
    double lambda, // regularization parameter
    int NumFeatures,
    Kernel kernel,
    skylark::ml::quasi_feature_transform_tag tag,
    int NumFeaturePartitions) :
    featureMaps(NumFeaturePartitions),
    NumFeatures(NumFeatures), NumFeaturePartitions(NumFeaturePartitions),
    loss(loss), regularizer(regularizer),
    starts(NumFeaturePartitions), finishes(NumFeaturePartitions),
    NumThreads(1), lambda(lambda), RHO(1.0), MAXITER(1000), TOL(0.1) {

    skylark::base::leaped_halton_sequence_t<value_type>
        qmcseq(kernel.qrft_sequence_dim()); // TODO size
    int cstart = 0, nf = NumFeatures, np = NumFeaturePartitions;
    for(int i = 0; i < NumFeaturePartitions; i++) {
        int sj = int(floor(double(nf) / np));
        starts[i] = cstart;
        finishes[i] = cstart + sj - 1;
        cstart += sj;
        nf -= sj;
        np--;

        featureMaps[i] =
            kernel.template create_qrft< data_matrix_t, local_matrix_t,
              skylark::base::leaped_halton_sequence_t>(sj, qmcseq,
                  starts[i], context);
    }
    this->ScaleFeatureMaps = true;
    OwnFeatureMaps = true;
    InitializeFactorizationCache();
    CacheTransforms = false;
}

// Guru interface
template <class InputType>
BlockADMMSolver<InputType>::BlockADMMSolver(
    const skylark::algorithms::loss_t<value_type>* loss,
    const skylark::algorithms::regularizer_t<value_type>* regularizer,
    const feature_transform_array_t &featureMaps,
    double lambda,
    bool ScaleFeatureMaps) :
    featureMaps(featureMaps),
    NumFeaturePartitions(featureMaps.size()),
    loss(loss), regularizer(regularizer),
    starts(NumFeaturePartitions), finishes(NumFeaturePartitions),
    NumThreads(1), lambda(lambda), RHO(1.0), MAXITER(1000), TOL(0.1)  {

    NumFeaturePartitions = featureMaps.size();
    NumFeatures = 0;
    for(int i = 0; i < NumFeaturePartitions; i++) {
        starts[i] = NumFeatures;
        finishes[i] = NumFeatures + featureMaps[i]->get_S() - 1;
        NumFeatures += featureMaps[i]->get_S();
    }
    this->ScaleFeatureMaps = ScaleFeatureMaps;
    OwnFeatureMaps = false;
    InitializeFactorizationCache();
    CacheTransforms = false;
}

template <class InputType>
BlockADMMSolver<InputType>::~BlockADMMSolver() {
    for(int i=0; i  < NumFeaturePartitions; i++) {
        delete Cache[i];
        if (OwnFeatureMaps)
            delete featureMaps[i];
    }
    delete[] Cache;
}


namespace internal {

template<typename T>
void GetSlice(El::Matrix<T> &X, El::Matrix<T> &Z,
    El::Int i, El::Int j, El::Int height, El::Int width) {

    El::View(Z, X, i, j, height, width);
}

template<typename T>
void GetSlice(skylark::base::sparse_matrix_t<T> &X, El::Matrix<T> &Z,
    El::Int i, El::Int j, El::Int height, El::Int width) {

    skylark::base::DenseSubmatrixCopy(X, Z, i, j, height, width);
}

}

template <class InputType>
skylark::ml::hilbert_model_t* BlockADMMSolver<InputType>::train(data_matrix_t& X, target_matrix_t& Y,
    data_matrix_t& Xv, target_matrix_t& Yv,
    bool regression, const boost::mpi::communicator& comm) {

    int rank = comm.rank();
    int size = comm.size();

    int P = size;

    int ni = skylark::base::Width(X);
    int d = skylark::base::Height(X);
    int targets = regression ? 1 : GetNumTargets(comm, Y);

    skylark::ml::hilbert_model_t* model =
        new skylark::ml::hilbert_model_t(featureMaps,
            ScaleFeatureMaps, NumFeatures, targets, regression);

    local_matrix_t Wbar;
    El::View(Wbar, model->get_coef());


    int k = Wbar.Width();

    // number of classes, targets - to generalize
    int D = NumFeatures;

    // exception: check if D = Wbar.Height();

    local_matrix_t O(k, ni);
    El::Zero(O);

    local_matrix_t Obar(k, ni);
    El::Zero(Obar);

    local_matrix_t nu(k, ni);
    El::Zero(nu);

    local_matrix_t W, mu, Wi, mu_ij, ZtObar_ij;

    if(rank==0) {
        El::Zeros(W,  D, k);
        El::Zeros(mu, D, k);
    }
    El::Zeros(Wi, D, k);
    El::Zeros(mu_ij, D, k);
    El::Zeros(ZtObar_ij, D, k);

    int iter = 0;

    value_type localloss = loss->evaluate(O, Y);
    value_type totalloss, accuracy, obj;

    int Dk = D*k;
    int nik  = ni*k;
    int start, finish, sj;

    boost::mpi::timer timer;

    local_matrix_t sum_o, del_o, wbar_output;
    El::Zeros(del_o, k, ni);
    local_matrix_t Yp(Yv.Height(), k);
    local_matrix_t Yp_labels(Yv.Height(), 1);

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
        broadcast(comm, Wbar.Buffer(), Dk, 0);

        SKYLARK_TIMER_ACCUMULATE(COMMUNICATION_PROFILE)

            // mu_ij = mu_ij - Wbar
            El::Axpy(-1.0, Wbar, mu_ij);

        // Obar = Obar - nu
        El::Axpy(-1.0, nu, Obar);

        SKYLARK_TIMER_RESTART(PROXLOSS_PROFILE);
        loss->proxoperator(Obar, 1.0/RHO, Y, O);
        SKYLARK_TIMER_ACCUMULATE(PROXLOSS_PROFILE);

        if(rank==0) {
            regularizer->proxoperator(Wbar, lambda/RHO, mu, W);
        }

        El::Zeros(sum_o, k, ni);
        El::Zeros(wbar_output, k, ni);

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

            local_matrix_t Z;

            // Get the Z matrix
            if (CacheTransforms && (iter > 1))
                El::View(Z, *TransformCache[j], 0, 0, sj, ni);
            else {
                if (featureMaps.size() > 0) {
                    featureMap = featureMaps[j];

                    SKYLARK_TIMER_RESTART(ZTRANSFORM_PROFILE);
                    Z.Resize(sj, ni); // TODO do we need this?
                    featureMap->apply(X, Z, skylark::sketch::columnwise_tag());
                    SKYLARK_TIMER_ACCUMULATE(ZTRANSFORM_PROFILE);

                    if (ScaleFeatureMaps)
                        El::Scale(sqrt(double(sj) / d), Z);
                } else
                    internal::GetSlice(X, Z, start, 0, sj, ni);
            }

            local_matrix_t tmp(sj, k);
            local_matrix_t rhs(sj, k);
            local_matrix_t o(k, ni);

            if(iter==1) {

                local_matrix_t Ones;
                El::Ones(Ones, sj, 1);
                El::Gemm(El::NORMAL, El::TRANSPOSE, 1.0, Z, Z, 0.0, *Cache[j]);
                El::UpdateDiagonal(*Cache[j], 1.0, Ones);
                El::Inverse(*Cache[j]);

                if (CacheTransforms)
                    *TransformCache[j] = Z;
            }

            El::View(tmp, Wbar, start, 0, sj, k); //tmp = Wbar[J,:]

            local_matrix_t wbar_tmp;
            El::Zeros(wbar_tmp, k, ni);

            if (NumThreads > 1) {
                El::Gemm(El::TRANSPOSE, El::NORMAL, 1.0, tmp, Z, 0.0, wbar_tmp);

#               ifdef SKYLARK_HAVE_OPENMP
#               pragma omp critical
#               endif
                El::Axpy(1.0, wbar_tmp, wbar_output);
            } else
                El::Gemm(El::TRANSPOSE, El::NORMAL, 1.0, tmp, Z, 1.0, wbar_output);

            rhs = tmp; //rhs = Wbar[J,:]
            El::View(tmp, mu_ij, start, 0, sj, k); //tmp = mu_ij[J,:]
            El::Axpy(-1.0, tmp, rhs); // rhs = rhs - mu_ij[J,:] = Wbar[J,:] - mu_ij[J,:]
            El::View(tmp, ZtObar_ij, start, 0, sj, k);
            El::Axpy(+1.0, tmp, rhs); // rhs = rhs + ZtObar_ij[J,:]

            SKYLARK_TIMER_RESTART(ZMULT_PROFILE);
            local_matrix_t dsum = del_o;
            El::Axpy(NumFeaturePartitions + 1.0, nu, dsum);
            El::Gemm(El::NORMAL, El::TRANSPOSE, 
                1.0/(NumFeaturePartitions + 1.0), Z, dsum, 1.0, rhs); // rhs = rhs + z'*(1/(n+1) * del_o + nu)
            SKYLARK_TIMER_ACCUMULATE(ZMULT_PROFILE);

            El::View(tmp, Wi, start, 0, sj, k);
            El::Gemm(El::NORMAL, El::NORMAL, 1.0, *Cache[j], rhs, 0.0, tmp); // ]tmp = Wi[J,:] = Cache[j]*rhs

            SKYLARK_TIMER_RESTART(ZMULT_PROFILE);
            El::Gemm(El::TRANSPOSE, El::NORMAL, 1.0, tmp, Z, 0.0, o); // o = (z*tmp)' = (z*Wi[J,:])'
            SKYLARK_TIMER_ACCUMULATE(ZMULT_PROFILE);

            // mu_ij[JJ,:] = mu_ij[JJ,:] + Wi[JJ,:];
            El::View(tmp, mu_ij, start, 0, sj, k); //tmp = mu_ij[J,:]
            El::View(rhs, Wi, start, 0, sj, k);
            El::Axpy(+1.0, rhs, tmp);

            //ZtObar_ij[JJ,:] = numpy.dot(Z.T, o);
            El::View(tmp, ZtObar_ij, start, 0, sj, k);
            El::Gemm(El::NORMAL, El::TRANSPOSE, 1.0, Z, o, 0.0, tmp);

            //  sum_o += o
            if (NumThreads > 1) {
#               ifdef SKYLARK_HAVE_OPENMP
#               pragma omp critical
#               endif
                El::Axpy(1.0, o, sum_o);
            } else
                El::Axpy(1.0, o, sum_o);

            Z.Empty(); // TODO do we need this?
        }

        SKYLARK_TIMER_ACCUMULATE(TRANSFORM_PROFILE);

        localloss = 0.0 ;
        //  El::Zeros(o, ni, k);
        local_matrix_t o(k, ni);
        El::Zero(o);
        El::Scale(-1.0, sum_o);
        El::Axpy(+1.0, O, sum_o); // sum_o = O.Matrix - sum_o
        del_o = sum_o;

        SKYLARK_TIMER_RESTART(PREDICTION_PROFILE);
        if (skylark::base::Width(Xv) > 0) {
            El::Zero(Yp);
            El::Zero(Yp_labels);
            model->predict(Xv, Yp_labels, Yp, NumThreads);

            if (regression) {
                El::Axpy(-1.0, Yv, Yp);
                value_type localerr = std::pow(El::Nrm2(Yp), 2);
                value_type localnrm = std::pow(El::Nrm2(Yv), 2);
                value_type err, nrm;
                boost::mpi::reduce(comm, localerr, err,
                    std::plus<value_type>(), 0);
                boost::mpi::reduce(comm, localnrm, nrm,
                    std::plus<value_type>(), 0);

                if (comm.rank() == 0)
                    accuracy = std::sqrt(err / nrm);
            } else {
                El::Int correct = skylark::ml::classification_accuracy(Yv, Yp);
                El::Int totalcorrect, total;
                boost::mpi::reduce(comm, correct, totalcorrect,
                    std::plus<El::Int>(), 0);
                boost::mpi::reduce(comm, Yv.Height(), total,
                    std::plus<El::Int>(), 0);

                if(comm.rank() == 0)
                    accuracy =  totalcorrect * 100.0 / total;
            }
        }
        SKYLARK_TIMER_ACCUMULATE(PREDICTION_PROFILE);

        localloss += loss->evaluate(wbar_output, Y);

        SKYLARK_TIMER_RESTART(COMMUNICATION_PROFILE);
        reduce(comm, localloss, totalloss, std::plus<value_type>(), 0);
        SKYLARK_TIMER_ACCUMULATE(COMMUNICATION_PROFILE);

        if(rank == 0) {
            obj = totalloss + lambda * regularizer->evaluate(Wbar);

            if (skylark::base::Width(Xv) <=0) {
                std::cout << "iteration " << iter
                          << " objective " << obj
                          << " time " << timer.elapsed()
                          << " seconds" << std::endl;
            }
            else {
                std::cout << "iteration " << iter
                          << " objective " << obj
                          << " accuracy " << boost::format("%.2f") % accuracy
                          << " time " << timer.elapsed()
                          << " seconds" << std::endl;
            }
        }

        El::Copy(O, Obar);
        El::Scale(1.0/(NumFeaturePartitions+1.0), sum_o);
        El::Axpy(-1.0, sum_o, Obar);

        El::Axpy(+1.0, O, nu);
        El::Axpy(-1.0, Obar, nu);

        SKYLARK_TIMER_RESTART(COMMUNICATION_PROFILE);
        boost::mpi::reduce (comm,
            Wi.LockedBuffer(),
            Wi.MemorySize(),
            Wbar.Buffer(),
            std::plus<value_type>(),
            0);
        SKYLARK_TIMER_ACCUMULATE(COMMUNICATION_PROFILE);

        if(rank==0) {
            //Wbar = (Wisum + W)/(P+1)
            El::Axpy(1.0, W, Wbar);
            El::Scale(1.0/(P+1), Wbar);

            // mu = mu + W - Wbar;
            El::Axpy(+1.0, W, mu);
            El::Axpy(-1.0, Wbar, mu);
        }

        SKYLARK_TIMER_RESTART(BARRIER_PROFILE);
        comm.barrier();
        SKYLARK_TIMER_ACCUMULATE(BARRIER_PROFILE);

        SKYLARK_TIMER_ACCUMULATE(ITERATIONS_PROFILE);
    }

    SKYLARK_TIMER_PRINT(ITERATIONS_PROFILE, comm);
    SKYLARK_TIMER_PRINT(COMMUNICATION_PROFILE, comm);
    SKYLARK_TIMER_PRINT(TRANSFORM_PROFILE, comm);
    SKYLARK_TIMER_PRINT(ZTRANSFORM_PROFILE, comm);
    SKYLARK_TIMER_PRINT(ZMULT_PROFILE, comm);
    SKYLARK_TIMER_PRINT(PROXLOSS_PROFILE, comm);
    SKYLARK_TIMER_PRINT(BARRIER_PROFILE, comm);
    SKYLARK_TIMER_PRINT(PREDICTION_PROFILE, comm);

    return model;
}


#endif /* SKYLARK_BLOCKADDM_HPP */
