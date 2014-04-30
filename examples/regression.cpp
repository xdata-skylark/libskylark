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
const int m = 2000;
const int n = 10;
const int t = 500;

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

template<>
template<typename KT>
struct exact_solver_type< skyalg::iterative_l2_solver_tag<KT> >:
    public skyalg::exact_regressor_t<
    regression_problem_type, rhs_type, sol_type, 
    skyalg::iterative_l2_solver_tag<KT> > {

    typedef skyalg::exact_regressor_t<
        regression_problem_type, rhs_type, sol_type, 
        skyalg::iterative_l2_solver_tag<KT> > base_type;

    exact_solver_type(const regression_problem_type& problem) :
        base_type(problem) {

    }

    exact_solver_type(const regression_problem_type& problem,
                      const skynla::precond_t<sol_type>& R) :
        base_type(problem, R) {

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
    skyalg::qr_l2_solver_tag,
    skyalg::sketch_and_solve_tag> {

    typedef skyalg::sketched_regressor_t<
        regression_problem_type, matrix_type, sol_type,
        skyalg::linear_tag,
        sketch_type,
        sketch_type,
        TransformType,
        skyalg::qr_l2_solver_tag,
        skyalg::sketch_and_solve_tag> base_type;

    sketched_solver_type(const regression_problem_type& problem,
        int sketch_size,
        skybase::context_t& context) :
        base_type(problem, sketch_size, context) {

    }

};

template<typename ProblemType, typename RhsType, typename SolType>
void check_solution(const ProblemType &pr, const RhsType &b, const SolType &x,
    double &res, double &resAtr) {
    RhsType r(b);
    skybase::Gemv(elem::NORMAL, -1.0, pr.input_matrix, x, 1.0, r);
    res = skybase::Nrm2(r);

    SolType Atr(x.Height(), x.Width(), x.Grid());
    skybase::Gemv(elem::TRANSPOSE, 1.0, pr.input_matrix, r, 0.0, Atr);
    resAtr = skybase::Nrm2(Atr);
}

int main(int argc, char** argv) {
    double res, resAtr;

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

    // Using exact regressor
    sol_type x(n,1);
    exact_solver_type<skyalg::qr_l2_solver_tag> exact_solver(problem);
    exact_solver.solve(b, x);
    check_solution(problem, b, x, res, resAtr);
    if (rank == 0)
        std::cout << "Exact (QR): ||r||_2 =  " << boost::format("%.2f") % res
                  << " ||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << std::endl;
    double res_opt = res;

    elem::Zero(x);
    exact_solver_type<
        skyalg::iterative_l2_solver_tag<
            skyalg::lsqr_tag > >(problem)
        .solve(b, x);
    check_solution(problem, b, x, res, resAtr);
    if (rank == 0)
        std::cout << "Exact (LSQR): ||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " ||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << std::endl;

    // Using sketch-and-solve
    sketched_solver_type<skysk::JLT_t>(problem, t, context).solve(b, x);
    check_solution(problem, b, x, res, resAtr);
    if (rank == 0)
        std::cout << "Sketch-and-Solve (JLT): ||r||_2 =  " 
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << " ||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << std::endl;

    sketched_solver_type<skysk::CWT_t>(problem, t, context).solve(b, x);
    check_solution(problem, b, x, res, resAtr);
    if (rank == 0)
        std::cout << "Sketch-and-Solve (CWT): ||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << " ||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << std::endl;

    sketched_solver_type<skysk::FJLT_t>(problem, t, context).solve(b, x);
    check_solution(problem, b, x, res, resAtr);
    if (rank == 0)
        std::cout << "Sketch-and-Solve (FJLT): ||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " (x " << boost::format("%.5f") % (res / res_opt) << ")"
                  << " ||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << std::endl;

    // Accelerate-using-sketching
    skysk::FJLT_t<matrix_type, sketch_type> S(m, t, context);
    sketch_type SA(t, n);
    S.apply(A, SA, skysk::columnwise_tag());
    precond_type R(n, n);
    elem::qr::Explicit(SA.Matrix(), R.Matrix());

    elem::Zero(x);
    skynla::tri_inverse_precond_t<sol_type, precond_type,
                                  elem::UPPER, elem::NON_UNIT> PR(R);
    exact_solver_type<
        skyalg::iterative_l2_solver_tag<
            skyalg::lsqr_tag > >(problem, PR)
        .solve(b, x);
    check_solution(problem, b, x, res, resAtr);
    if (rank == 0)
        std::cout << "Accelerate-using-sketching (FJLT, LSQR): ||r||_2 =  "
                  << boost::format("%.2f") % res
                  << " ||A' * r||_2 = " << boost::format("%.2e") % resAtr
                  << std::endl;
    return 0;
}
