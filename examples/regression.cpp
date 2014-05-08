#include <iostream>

#include <elemental.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <skylark.hpp>

/*******************************************/
namespace bmpi =  boost::mpi;
namespace skybase = skylark::base;
namespace skysk =  skylark::sketch;
namespace skynla = skylark::nla;
namespace skyalg = skylark::algorithms;
namespace skyutil = skylark::utility;
/*******************************************/

// Parameters
#if 0
const int m = 2000;
const int n = 10;
const int t = 500;
#else
const int m = 30000;
const int n = 500;
const int t = 2000;
#endif

typedef elem::DistMatrix<double, elem::VC, elem::STAR> matrix_type;
typedef elem::DistMatrix<double, elem::VC, elem::STAR> rhs_type;
typedef elem::DistMatrix<double, elem::STAR, elem::STAR> sol_type;
typedef elem::DistMatrix<double, elem::STAR, elem::STAR> sketch_type;
typedef elem::DistMatrix<double, elem::STAR, elem::STAR> precond_type;

typedef skyalg::regression_problem_t<matrix_type,
                                     skyalg::linear_tag,
                                     skyalg::l2_tag,
                                     skyalg::no_reg_tag> regression_problem_type;

template<typename AlgTag>
struct exact_solver_type :
    public skyalg::exact_regressor_t<
    regression_problem_type, rhs_type, sol_type, AlgTag> {

    typedef skyalg::exact_regressor_t<
        regression_problem_type, rhs_type, sol_type, AlgTag> base_type;

    exact_solver_type(const regression_problem_type& problem) :
        base_type(problem) {

    }
};

template<template <typename, typename> class TransformType >
struct fast_exact_solver_type_sb :
    public skyalg::fast_exact_regressor_t<
    regression_problem_type, rhs_type, sol_type,
    skyalg::simplified_blendenpik_tag<TransformType,
                                      skyalg::svd_precond_tag> > {

    typedef  skyalg::fast_exact_regressor_t<
        regression_problem_type, rhs_type, sol_type,
        skyalg::simplified_blendenpik_tag<TransformType,
                                          skyalg::svd_precond_tag > > base_type;

    fast_exact_solver_type_sb(const regression_problem_type& problem,
        skybase::context_t& context) :
        base_type(problem, context) {

    }
};

struct fast_exact_solver_type_lsrn :
    public skyalg::fast_exact_regressor_t<
    regression_problem_type, rhs_type, sol_type,
    skyalg::lsrn_tag<skyalg::qr_precond_tag> > {

    typedef  skyalg::fast_exact_regressor_t<
        regression_problem_type, rhs_type, sol_type,
        skyalg::lsrn_tag<skyalg::qr_precond_tag > > base_type;

    fast_exact_solver_type_lsrn(const regression_problem_type& problem,
        skybase::context_t& context) :
        base_type(problem, context) {

    }
};

template<>
template<typename KT>
struct exact_solver_type< skyalg::iterative_l2_solver_tag<KT> >:
    public skyalg::exact_regressor_t<
    regression_problem_type, rhs_type, sol_type,
    skyalg::iterative_l2_solver_tag<KT> > {

    typedef skyalg::exact_regressor_t<
        regression_problem_type, rhs_type, sol_type,
        skyalg::iterative_l2_solver_tag<KT> > base_type;

    exact_solver_type(const regression_problem_type& problem,
        skynla::iter_params_t iter_params) :
        base_type(problem, iter_params) {

    }

    exact_solver_type(const regression_problem_type& problem,
        const skynla::precond_t<sol_type>& R,
        skynla::iter_params_t iter_params) :
        base_type(problem, R, iter_params) {

    }


};

template<template <typename, typename> class TransformType >
struct sketched_solver_type :
    public skyalg::sketched_regressor_t<
    regression_problem_type, matrix_type, sol_type,
    skyalg::linear_tag,
    sketch_type,
    sketch_type,
    TransformType,
    skyalg::qr_l2_solver_tag> {

    typedef skyalg::sketched_regressor_t<
        regression_problem_type, matrix_type, sol_type,
        skyalg::linear_tag,
        sketch_type,
        sketch_type,
        TransformType,
        skyalg::qr_l2_solver_tag> base_type;

    sketched_solver_type(const regression_problem_type& problem,
        int sketch_size,
        skybase::context_t& context) :
        base_type(problem, sketch_size, context) {

    }

};

template<typename ProblemType, typename RhsType, typename SolType>
void check_solution(const ProblemType &pr, const RhsType &b, const SolType &x, 
    const RhsType &r0,
    double &res, double &resAtr, double &resFac) {
    RhsType r(b);
    skybase::Gemv(elem::NORMAL, -1.0, pr.input_matrix, x, 1.0, r);
    res = skybase::Nrm2(r);

    SolType Atr(x.Height(), x.Width(), x.Grid());
    skybase::Gemv(elem::TRANSPOSE, 1.0, pr.input_matrix, r, 0.0, Atr);
    resAtr = skybase::Nrm2(Atr);

    skybase::Axpy(-1.0, r0, r);
    RhsType dr(b);
    skybase::Axpy(-1.0, r0, dr);
    resFac = skybase::Nrm2(r) / skybase::Nrm2(dr);
}

int main(int argc, char** argv) {
    double res, resAtr, resFac;

    bmpi::environment env(argc, argv);
    bmpi::communicator world;

    elem::Initialize(argc, argv);
    MPI_Comm mpi_world(world);
    elem::Grid grid(mpi_world);
    int rank = world.rank();

    skybase::context_t context(23234);

    // Setup problem and righthand side
    // Using Skylark's uniform generator (as opposed to Elemental's)
    // will insure the same A and b are generated regardless of the number
    // of processors.
    matrix_type A =
        skyutil::uniform_matrix_t<matrix_type>::generate(m,
            n, elem::DefaultGrid(), context);
    matrix_type b =
        skyutil::uniform_matrix_t<matrix_type>::generate(m,
            1, elem::DefaultGrid(), context);

    regression_problem_type problem(m, n, A);

    boost::mpi::timer timer;
    double telp;

    sol_type x(n,1);

    rhs_type r(b);

    // Using QR
    timer.restart();
    exact_solver_type<skyalg::qr_l2_solver_tag> exact_solver(problem);
    exact_solver.solve(b, x);
    telp = timer.elapsed();
    check_solution(problem, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "Exact (QR):\t\t\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << "\t\t\t\t\t\t\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;
    double res_opt = res;

    skybase::Gemv(elem::NORMAL, -1.0, problem.input_matrix, x, 1.0, r);

    // Using LSQR
    skynla::iter_params_t lsqrparams;
    lsqrparams.am_i_printing = rank == 0;
    lsqrparams.log_level = 0;
    timer.restart();
    exact_solver_type<
        skyalg::iterative_l2_solver_tag<
            skyalg::lsqr_tag > >(problem, lsqrparams)
        .solve(b, x);
    telp = timer.elapsed();
    check_solution(problem, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "Exact (LSQR):\t\t\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << "\t\t||r - r*||_2 / ||b - r*||_2 = " << boost::format("%.2e") % resFac
                  << "\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;

    // Using sketch-and-solve

#if 0 
    timer.restart();
    sketched_solver_type<skysk::JLT_t>(problem, t, context).solve(b, x);
    telp = timer.elapsed();
    check_solution(problem, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "Sketch-and-Solve (JLT):\t\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << "\t||r - r*||_2 / ||b - r*||_2 = " << boost::format("%.2e") % resFac
                  << "\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;
#endif

    timer.restart();
    sketched_solver_type<skysk::CWT_t>(problem, t, context).solve(b, x);
    telp = timer.elapsed();
    check_solution(problem, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "Sketch-and-Solve (CWT):\t\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << "\t||r - r*||_2 / ||b - r*||_2 = " << boost::format("%.2e") % resFac
                  << "\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;

    timer.restart();
    sketched_solver_type<skysk::FJLT_t>(problem, t, context).solve(b, x);
    telp = timer.elapsed();
    check_solution(problem, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "Sketch-and-Solve (FJLT):\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << "\t||r - r*||_2 / ||b - r*||_2 = " << boost::format("%.2e") % resFac
                  << "\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;

    // Accelerate-using-sketching
#if 0
    timer.restart();
    fast_exact_solver_type_sb<skysk::JLT_t>(problem, context).solve(b, x);
    telp = timer.elapsed();
    check_solution(problem, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "Simplified Blendenpik (JLT):\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << "\t||r - r*||_2 / ||b - r*||_2 = " << boost::format("%.2e") % resFac
                  << "\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;
#endif

    timer.restart();
    fast_exact_solver_type_sb<skysk::FJLT_t>(problem, context).solve(b, x);
    telp = timer.elapsed();
    check_solution(problem, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "Simplified Blendenpik (FJLT):\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << "\t||r - r*||_2 / ||b - r*||_2 = " << boost::format("%.2e") % resFac
                  << "\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;

    timer.restart();
    fast_exact_solver_type_sb<skysk::CWT_t>(problem, context).solve(b, x);
    telp = timer.elapsed();
    check_solution(problem, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "Simplified Blendenpik (CWT):\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << "\t||r - r*||_2 / ||b - r*||_2 = " << boost::format("%.2e") % resFac
                  << "\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;

    timer.restart();
    fast_exact_solver_type_lsrn(problem, context).solve(b, x);
    telp = timer.elapsed();
    check_solution(problem, b, x, r, res, resAtr, resFac);
    if (rank == 0)
        std::cout << "LSRN:\t\t\t\t||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << "\t||r - r*||_2 / ||b - r*||_2 = " << boost::format("%.2e") % resFac
                  << "\t||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << "\t\tTime: " << boost::format("%.2e") % telp << " sec"
                  << std::endl;

    return 0;
}
