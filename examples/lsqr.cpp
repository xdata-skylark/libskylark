/**
 * This file cannot be built without CombBLAS and Elemental both being present.
 * The accompanying CMakeLists.txt checks for both libraries to be available.
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <vector>
#include <utility>
#include <ext/hash_map>

#include <elemental.hpp>
#include <CombBLAS.h>
#include <boost/mpi.hpp>
#include <skylark.hpp>

#include "utilities.hpp"
#include "parser.hpp"

/*******************************************/
namespace bmpi =  boost::mpi;
namespace skys =  skylark::sketch;
namespace skyb =  skylark::base;
namespace skyalg = skylark::algorithms;
namespace skynla = skylark::nla;
namespace skyutil = skylark::utility;
/*******************************************/

/* These were declared as extern in utilities.hpp --- defining it here */
int int_params [NUM_INT_PARAMETERS];
char* chr_params[NUM_CHR_PARAMETERS];

/** Typedef DistMatrix and Matrix */
typedef std::vector<int> IntContainer;
typedef std::vector<double> DblContainer;
typedef elem::Matrix<double> DenseMatrixType;
typedef elem::DistMatrix<double> DenseDistMatrixType;

typedef SpDCCols<int, double> SparseMatrixType;
typedef SpParMat<int, double, SparseMatrixType> SparseDistMatrixType;
typedef FullyDistVec<int, double> SparseVectorType;
typedef FullyDistMultiVec<int, double> SparseMultiVectorType;

typedef skyutil::uniform_matrix_t<DenseDistMatrixType> uni_dense_dist_mat_t;
typedef skyutil::uniform_matrix_t<SparseDistMatrixType> uni_sparse_dist_mat_t;
typedef skyutil::uniform_matrix_t<SparseMultiVectorType> 
                                                  uni_sparse_dist_multi_vec_t;

typedef skyutil::empty_matrix_t<DenseDistMatrixType> empty_dense_dist_mat_t;
typedef skyutil::empty_matrix_t<SparseDistMatrixType> empty_sparse_dist_mat_t;
typedef skyutil::empty_matrix_t<SparseMultiVectorType> 
                                                  empty_sparse_dist_multi_vec_t;

typedef skyutil::print_t<SparseDistMatrixType> sparse_dist_mat_printer_t;
typedef skyutil::print_t<SparseMultiVectorType> sparse_dist_multi_vec_printer_t;
typedef skyutil::print_t<DenseDistMatrixType> dense_dist_mat_printer_t;

int main (int argc, char** argv) {
  /* Initialize MPI */
  bmpi::environment env (argc, argv);

  /* Create a global communicator */
  bmpi::communicator world;

  /* MPI sends argc and argv everywhere --- parse everywhere */
  parse_parameters (argc,argv);

  /* Initialize skylark */
  skyb::context_t context (int_params[RAND_SEED_INDEX]);

  /** Only left sketch is supported for now */
  if (SKETCH_LEFT != int_params[SKETCH_DIRECTION_INDEX]) {
    std::cout << "We don't support reading --- yet --" << std::endl;
    goto END;
  }

  /** Only randomization is supported for now */
  if (0==int_params[USE_RANDOM_INDEX]) {
    /** TODO: Read the entries! */
    std::cout << "We don't support reading --- yet --" << std::endl;
    goto END;
  }

  if (0==strcmp("elem",chr_params[MATRIX_TYPE_INDEX])) {
    /** Run the test for elemental matrices */

    /* Initialize elemental */
    elem::Initialize (argc, argv);
    MPI_Comm mpi_world(world);
    elem::Grid grid (mpi_world);
  
    DenseDistMatrixType A=uni_dense_dist_mat_t::generate
            (int_params[M_INDEX], int_params[N_INDEX], grid, context);
    DenseDistMatrixType B=uni_dense_dist_mat_t::generate
            (int_params[M_INDEX], int_params[N_RHS_INDEX], grid, context);
    DenseDistMatrixType X(int_params[N_INDEX], int_params[N_RHS_INDEX], grid);

    /** Depending on which sketch is requested, do the sketching */
    DenseMatrixType sketch_A(int_params[S_INDEX], int_params[N_INDEX]);
    DenseMatrixType sketch_B(int_params[S_INDEX], int_params[N_RHS_INDEX]);
    DenseMatrixType sketch_X(int_params[N_INDEX], int_params[N_RHS_INDEX]);

    /***********************************************************************/
    /* Hack to make sure that we can sketch matrices with MC,MR distributn */
    /***********************************************************************/
    typedef elem::DistMatrix<double, elem::VC, elem::STAR> CheatDistMatrixType;
    CheatDistMatrixType A_cheat(A), B_cheat(B);
    /***********************************************************************/
    if (0==strcmp("JLT", chr_params[TRANSFORM_INDEX]) ) {
      skys::JLT_t<CheatDistMatrixType, DenseMatrixType> 
        JLT (int_params[M_INDEX], int_params[S_INDEX], context);
      JLT.apply (A_cheat, sketch_A, skys::columnwise_tag());
      JLT.apply (B_cheat, sketch_B, skys::columnwise_tag());
    } else if (0==strcmp("FJLT", chr_params[TRANSFORM_INDEX]) ) {
      skys::FJLT_t<CheatDistMatrixType, DenseMatrixType> 
        FJLT (int_params[M_INDEX], int_params[S_INDEX], context);
        FJLT.apply (A_cheat, sketch_A, skys::columnwise_tag());
        FJLT.apply (B_cheat, sketch_B, skys::columnwise_tag());
    } else if (0==strcmp("CWT", chr_params[TRANSFORM_INDEX]) ) {
      skys::CWT_t<CheatDistMatrixType, DenseMatrixType>
        Sparse (int_params[M_INDEX], int_params[S_INDEX], context);
      Sparse.apply (A_cheat, sketch_A, skys::columnwise_tag());
      Sparse.apply (B_cheat, sketch_B, skys::columnwise_tag());
    } else {
      std::cout << "We only have JLT/FJLT/CWT sketching" << std::endl;
    }
    A_cheat.Empty();
    B_cheat.Empty();

    /** Set up the iterative solver parameters */
    skynla::iter_params_t params(1e-14,  /* tolerance */
                                 world.rank()==0, /* only root prints */
                                 100, /* number of iterations */
                                 int_params[DEBUG_INDEX]-1); /* debugging */

    /** Solve the exact problem */
    DblContainer exact_norms(int_params[N_RHS_INDEX], 0.0);
    skyalg::regression_problem_t<skyalg::l2_tag, DenseDistMatrixType> 
            exact_problem(int_params[M_INDEX], int_params[N_INDEX], A);
    skyalg::exact_regressor_t
        <skyalg::l2_tag,
         DenseDistMatrixType, 
         DenseDistMatrixType,
         skyalg::iterative_l2_solver_tag<skyalg::lsqr_tag> > 
         exact_regr(exact_problem);
    exact_regr.solve(B, X, params);
    skynla::iter_solver_op_t<DenseDistMatrixType, DenseDistMatrixType>::
                      residual_norms(A, B, X, exact_norms);

    if (1<int_params[DEBUG_INDEX] && 
        (int_params[M_INDEX]*int_params[N_INDEX])<100 && 
        world.rank() == 0) {
      elem::Display(A, "A"); 
      elem::Display(B, "B");
      elem::Display(X, "X");
    }

    /** Solve the sketched problem---change to using the sketched regressor */
    DblContainer sketched_norms(int_params[N_RHS_INDEX], 0.0);
    skyalg::regression_problem_t<skyalg::l2_tag, DenseMatrixType> 
        sketched_problem(int_params[S_INDEX], int_params[N_INDEX], sketch_A);
    skyalg::exact_regressor_t
        <skyalg::l2_tag,
         DenseMatrixType, 
         DenseMatrixType,
         skyalg::iterative_l2_solver_tag<skyalg::lsqr_tag> > 
         sketched_regr(sketched_problem);
    sketched_regr.solve(sketch_B, sketch_X, params);
    DenseDistMatrixType sketch_X_global(X);
    skynla::iter_solver_op_t<DenseDistMatrixType, DenseDistMatrixType>::
                    make_dist_from_local (sketch_X, sketch_X_global);
    skynla::iter_solver_op_t<DenseDistMatrixType, DenseDistMatrixType>::
                      residual_norms(A, B, sketch_X_global, sketched_norms);

    if (1<int_params[DEBUG_INDEX] && 
        (int_params[M_INDEX]*int_params[N_INDEX])<100 && 
        world.rank() == 0) {
      dense_dist_mat_printer_t::apply (A, "A", true, 2);
      dense_dist_mat_printer_t::apply (B, "B", true, 2);
      dense_dist_mat_printer_t::apply (X, "X", true, 2);
    }

    /** Check the quality of the solution */
    if (0==world.rank()) {
      for (int i=0; i<int_params[N_RHS_INDEX]; ++i) 
        printf ("For RHS (%d), exact norm = %lf, sketched norm=%lf\n", 
                              i, exact_norms[i], sketched_norms[i]);
    }

    /** Finalize Elemental */
    elem::Finalize();

  } else {
    /** Run the test for CombBLAS matrices */

    /* Create matrices A and B */
    const int NNZ = int_params[M_INDEX]*int_params[N_INDEX]*0.5;
    SparseDistMatrixType A=uni_sparse_dist_mat_t::generate
            (int_params[M_INDEX], int_params[N_INDEX], NNZ, context);
    SparseMultiVectorType B=uni_sparse_dist_multi_vec_t::generate
            (int_params[M_INDEX], int_params[N_RHS_INDEX], context);
    SparseMultiVectorType X (int_params[N_INDEX], int_params[N_RHS_INDEX]);

    /** Depending on which sketch is requested, do the sketching */
    SparseDistMatrixType sketch_A=empty_sparse_dist_mat_t::generate
            (int_params[S_INDEX], int_params[N_INDEX]);
    SparseMultiVectorType sketch_B=empty_sparse_dist_multi_vec_t::generate
            (int_params[S_INDEX], int_params[N_RHS_INDEX]);
    SparseMultiVectorType sketch_X=empty_sparse_dist_multi_vec_t::generate
            (int_params[N_INDEX], int_params[N_RHS_INDEX]);
    if (0==strcmp("CWT", chr_params[TRANSFORM_INDEX]) ) {
      skys::CWT_t<SparseDistMatrixType, SparseDistMatrixType> 
        CWT_mat (int_params[M_INDEX], int_params[S_INDEX], context);
      CWT_mat.apply (A, sketch_A, skys::columnwise_tag());
      skys::CWT_t<SparseMultiVectorType, SparseMultiVectorType> 
        CWT_vec (CWT_mat);
      CWT_vec.apply (B, sketch_B, skys::columnwise_tag());
    } else {
      std::cout << "We only have CWT for sparse sketching" << std::endl;
    }

    /** Set up the iterative solver parameters */
    skynla::iter_params_t params(1e-14,  /* tolerance */
                                 world.rank()==0, /* only root prints */
                                 100, /* number of iterations */
                                 int_params[DEBUG_INDEX]-1); /* debugging */

    /** Solve the exact problem */
    DblContainer exact_norms(int_params[N_RHS_INDEX], 0.0);
    skyalg::regression_problem_t<skyalg::l2_tag, SparseDistMatrixType> 
            exact_problem(int_params[M_INDEX], int_params[N_INDEX], A);
    skyalg::exact_regressor_t
        <skyalg::l2_tag,
         SparseDistMatrixType, 
         SparseMultiVectorType,
         skyalg::iterative_l2_solver_tag<skyalg::lsqr_tag> > 
         exact_regr(exact_problem);
    exact_regr.solve(B, X, params);
    skynla::iter_solver_op_t<SparseDistMatrixType, SparseMultiVectorType>::
                      residual_norms(A, B, X, exact_norms);

    /** Solve the sketched problem */
    DblContainer sketched_norms(int_params[N_RHS_INDEX], 0.0);
    skyalg::regression_problem_t<skyalg::l2_tag, SparseDistMatrixType> 
         sketched_problem(int_params[S_INDEX], int_params[N_INDEX], sketch_A);
    skyalg::exact_regressor_t
        <skyalg::l2_tag,
         SparseDistMatrixType, 
         SparseMultiVectorType,
         skyalg::iterative_l2_solver_tag<skyalg::lsqr_tag> > 
         sketched_regr(sketched_problem);
    sketched_regr.solve(sketch_B, sketch_X, params);
    skynla::iter_solver_op_t<SparseDistMatrixType, SparseMultiVectorType>::
              residual_norms(A, B, sketch_X, sketched_norms);

    if (1<int_params[DEBUG_INDEX] && 
        (int_params[M_INDEX]*int_params[N_INDEX])<100 && 
        world.rank() == 0) {
      sparse_dist_mat_printer_t::apply (A, "A", true, 2);
      sparse_dist_multi_vec_printer_t::apply (B, "B", true, 2);
      sparse_dist_multi_vec_printer_t::apply (X, "X", true, 2);
    }

    /** Check the quality of the solution */
    if (0==world.rank()) {
      for (int i=0; i<int_params[N_RHS_INDEX]; ++i) 
        printf ("For RHS (%d), exact norm = %lf, sketched norm=%lf\n", 
                              i, exact_norms[i], sketched_norms[i]);
    }
  }

END:
  return 0;
}
