#include <iostream>

#include <elemental.hpp>
#include <boost/mpi.hpp>
#include <skylark.hpp>

/*******************************************/
namespace bmpi =  boost::mpi;
namespace skysk =  skylark::sketch;
namespace skyalg = skylark::algorithms;
/*******************************************/

typedef elem::DistMatrix<double> MatrixType1;
typedef elem::DistMatrix<double, elem::VC, elem::STAR> MatrixType;
typedef elem::Matrix<double> SketchType;

const int m = 2000;
const int n = 10;
const int t = 500;

// TODO move to NLA layer
template<typename T, elem::Distribution U, elem::Distribution V>
inline T Nrm2(const elem::DistMatrix<T, U, V>& x)
{
    return elem::FrobeniusNorm(x);
}

template<typename MatrixType>
void check_solution(
    skyalg::regression_problem_t<skyalg::l2_tag, MatrixType> pr,
    const MatrixType &b, const MatrixType &x,
    int rank) {
    MatrixType r(b.Grid());
    r = b;
    elem::Gemv(elem::NORMAL, -1.0, pr.input_matrix, x, 1.0, r);
    double res = elem::Nrm2(r);
    if (rank == 0)
        std::cout << "Residual for exact solve is " << res << std::endl;

    MatrixType Atr(x.Height(), x.Width(), x.Grid());
    elem::Gemv(elem::TRANSPOSE, 1.0, pr.input_matrix, r, 0.0, Atr);
    double resAtr = elem::Nrm2(Atr);
    if (rank == 0)
        std::cout << "For exact solve: ||A' * r||_2 = " << resAtr << std::endl;
}

int main(int argc, char** argv) {
    bmpi::environment env(argc, argv);
    bmpi::communicator world;

    elem::Initialize(argc, argv);
    MPI_Comm mpi_world(world);
    elem::Grid grid(mpi_world);
    int rank = world.rank();

    skysk::context_t context(23234, world);

    // Setup problem and righthand side
    MatrixType A(m, n);
    elem::MakeUniform(A);
    skyalg::regression_problem_t<skyalg::l2_tag, MatrixType> problem(m, n, A);

    MatrixType b(m, 1);
    elem::MakeUniform(b);

    // Using exact regressor
    MatrixType1 b1(m, 1);
    MatrixType1 x1(n, 1);
    MatrixType1 A1(m, n);
    A1 = A; b1 = b;
    skyalg::regression_problem_t<skyalg::l2_tag, MatrixType1> problem1(m, n, A1);
    skyalg::exact_regressor_t<skyalg::l2_tag,
                              MatrixType1, MatrixType1,
                              skyalg::qr_l2_solver_tag> exact_regr(problem1);
    exact_regr.solve(b1, x1);
    check_solution(problem1, b1, x1, rank);

    // Using sketch-and-solve
    SketchType x2(n, 1);
    skyalg::sketched_regressor_t<
        skyalg::l2_tag,
        MatrixType, MatrixType, SketchType,
        skysk::CWT_t,
        skyalg::qr_l2_solver_tag,
        skyalg::sketch_and_solve_tag> sketched_regr(problem, t, context);
    sketched_regr.solve(b, x2);
    MatrixType r(b.Grid());
    r = b;
    // TODO part to be moved to NLA
    elem::DistMatrix<double, elem::STAR, elem::STAR> xx(n, 1);
    if (rank == 0)
        xx.Matrix() = x2;
    boost::mpi::broadcast(world, xx.Buffer(), n, 0);
    elem::Gemv(elem::NORMAL,
        -1.0, problem.input_matrix.LockedMatrix(),
        xx.LockedMatrix(), 1.0,
        r.Matrix());
    double res = Nrm2(r);
    if (rank == 0)
        std::cout << "Sketched residual is " << res << std::endl;

    elem::Matrix<double> Atr_local(n, 1), Atr(n, 1);
    // TODO parts to be moved to NLA
    elem::Gemv(elem::TRANSPOSE,
        1.0, problem.input_matrix.LockedMatrix(),
        r.LockedMatrix(), 0.0, Atr_local);
    boost::mpi::reduce(world,
        Atr_local.LockedBuffer(),
        n,
        Atr.Buffer(),
        std::plus<double>(),
        0);
    if (rank == 0) {
        double resAtr = elem::Nrm2(Atr);
        std::cout << "Sketched ||A' * r||_2 = " << resAtr << std::endl;
    }

    return 0;
}
