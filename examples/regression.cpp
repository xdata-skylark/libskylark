#include <iostream>

#include <elemental.hpp>
#include <boost/mpi.hpp>
#include <skylark.hpp>

/*******************************************/
namespace bmpi =  boost::mpi;
namespace skybase = skylark::base;
namespace skysk =  skylark::sketch;
namespace skyalg = skylark::algorithms;
namespace skyutil = skylark::utility;
/*******************************************/

// Parameters
const int m = 2000;
const int n = 10;
const int t = 500;

#define SKETCH_TYPE skysk::JLT_t

typedef elem::DistMatrix<double> MatrixType1;
typedef elem::DistMatrix<double, elem::VC, elem::STAR> MatrixType;
typedef elem::DistMatrix<double, elem::VC, elem::STAR> RhsType;
typedef elem::DistMatrix<double, elem::STAR, elem::STAR> SolType;
typedef elem::DistMatrix<double, elem::STAR, elem::STAR> SketchType;

typedef skyalg::regression_problem_t<MatrixType,
                                     skyalg::linear_tag,
                                     skyalg::l2_tag,
                                     skyalg::no_reg_tag> RegressionProblemType;

typedef skyalg::exact_regressor_t<
    RegressionProblemType,
    RhsType,
    SolType,
    skyalg::qr_l2_solver_tag> ExactRegressorType;

typedef skyalg::sketched_regressor_t<
    RegressionProblemType, MatrixType, SolType,
    skyalg::linear_tag,
    SketchType,
    SketchType,
    SKETCH_TYPE,
    skyalg::qr_l2_solver_tag,
    skyalg::sketch_and_solve_tag> SketchedRegressorType;

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
    MatrixType A =
        skyutil::uniform_matrix_t<MatrixType>::generate(m,
            n, elem::DefaultGrid(), context);
    MatrixType b =
        skyutil::uniform_matrix_t<MatrixType>::generate(m,
            1, elem::DefaultGrid(), context);

    RegressionProblemType problem(m, n, A);

    // Using exact regressor
    SolType x(n,1);
    ExactRegressorType exact_regr(problem);
    exact_regr.solve(b, x);
    check_solution(problem, b, x, res, resAtr);
    if (rank == 0) {
        std::cout << "Residual for exact solve is " << res << std::endl;
        std::cout << "For exact solve: ||A' * r||_2 = " << resAtr << std::endl;
    }

    // Using sketch-and-solve
    SolType xx(n, 1);
    SketchedRegressorType sketched_regr(problem, t, context);
    sketched_regr.solve(b, xx);
    check_solution(problem, b, xx, res, resAtr);
    if (rank == 0) {
        std::cout << "Residual for sketched solve is " << res << std::endl;
        std::cout << "For sketched solve: ||A' * r||_2 = " << resAtr << std::endl;
    }
    return 0;
}
