/**
 *  This test ensures that the sketch application (for CombBLAS matrices) is
 *  done correctly (on-the-fly matrix multiplication in the code is compared
 *  to true matrix multiplication).
 *  This test builds on the following assumptions:
 *
 *      - CombBLAS PSpGEMM returns the correct result, and
 *      - the random numbers in row_idx and row_value (see
 *        hash_transform_data_t) are drawn from the promised distributions.
 */


#include <vector>

#include <boost/mpi.hpp>
#include <boost/test/minimal.hpp>

#include "../../utility/distributions.hpp"
#include "../../utility/sparse_matrix.hpp"
#include "../../sketch/context.hpp"
#include "../../sketch/hash_transform.hpp"


template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct Dummy_t : public skylark::sketch::hash_transform_t<
    InputMatrixType, OutputMatrixType,
    boost::random::uniform_int_distribution,
    skylark::utility::rademacher_distribution_t > {

    typedef skylark::sketch::hash_transform_t<
        InputMatrixType, OutputMatrixType,
        boost::random::uniform_int_distribution,
        skylark::utility::rademacher_distribution_t >
            hash_t;

    Dummy_t(int N, int S, skylark::sketch::context_t& context)
        : skylark::sketch::hash_transform_t<InputMatrixType, OutputMatrixType,
          boost::random::uniform_int_distribution,
          skylark::utility::rademacher_distribution_t>(N, S, context)
    {}

    std::vector<size_t> getRowIdx() { return hash_t::row_idx; }
    std::vector<double> getRowValues() { return hash_t::row_value; }
};

int test_main(int argc, char *argv[]) {

    //////////////////////////////////////////////////////////////////////////
    //[> Parameters <]

    //FIXME: use random sizes?
    const size_t n   = 10;
    const size_t m   = 5;
    const size_t n_s = 6;
    const size_t m_s = 3;

    typedef skylark::utility::sparse_matrix_t<size_t, double> Matrix_t;

    //////////////////////////////////////////////////////////////////////////
    //[> Setup test <]
    namespace mpi = boost::mpi;
    mpi::environment env(argc, argv);
    mpi::communicator world;
    const size_t rank = world.rank();

    skylark::sketch::context_t context (0, world);

    double count = 1.0;

    const size_t matrix_full = n * m;
    std::vector<int> rowsf(n + 1);
    std::vector<int> colsf(matrix_full);
    std::vector<double> valsf(matrix_full);

    for(size_t i = 0; i < n + 1; ++i)
        rowsf[i] = i * m;

    for(size_t i = 0; i < matrix_full; ++i) {
        colsf[i] = i % m;
        valsf[i] = count;
        count++;
    }

    Matrix_t A;
    A.Attach(&rowsf[0], &colsf[0], &valsf[0], n + 1, matrix_full);

    count = 1;
    size_t row = 0;
    typename Matrix_t::const_ind_itr_range_t ritr = A.indptr_itr();
    typename Matrix_t::const_ind_itr_range_t citr = A.indices_itr();
    typename Matrix_t::const_val_itr_range_t vitr = A.values_itr();

    for(; ritr.first + 1 != ritr.second; ritr.first++, ++row) {
        for(size_t idx = 0; idx < (*(ritr.first + 1) - *ritr.first);
            citr.first++, vitr.first++, ++idx) {

            BOOST_REQUIRE( *vitr.first == count );
            count++;
        }
    }


    //////////////////////////////////////////////////////////////////////////
    //[> Column wise application <]

    //[> 1. Create the sketching matrix <]
    Dummy_t<Matrix_t, Matrix_t> Sparse(n, n_s, context);
    std::vector<size_t> row_idx = Sparse.getRowIdx();
    std::vector<double> row_val = Sparse.getRowValues();

    // PI generated by random number gen
    size_t sketch_size = row_val.size();
    std::vector<int> rows(sketch_size);
    std::vector<int> cols(sketch_size);
    std::vector<double> vals(sketch_size);

    typename Matrix_t::coords_t coords;
    for(size_t i = 0; i < sketch_size; ++i) {
        typename Matrix_t::coord_tuple_t new_entry(row_idx[i], i, row_val[i]);
        coords.push_back(new_entry);
    }

    Matrix_t pi_sketch;
    pi_sketch.Attach(coords);

    //[> 2. Create sketched matrix <]
    Matrix_t sketch_A;

    //[> 3. Apply the transform <]
    Sparse.apply(A, sketch_A, skylark::sketch::columnwise_tag());

    //[> 4. Build structure to compare: PI * A ?= sketch_A <]

    row = 0;
    typename Matrix_t::coords_t coords_new;
    ritr = pi_sketch.indptr_itr();
    citr = pi_sketch.indices_itr();
    vitr = pi_sketch.values_itr();

    // multiply with vector where an entry has the value:
    //   col_idx + row_idx * m.
    // See creation of A.
    for(; ritr.first + 1 != ritr.second; ritr.first++, ++row) {
        for(size_t idx = 0; idx < (*(ritr.first + 1) - *ritr.first);
            citr.first++, vitr.first++, ++idx) {

            for(size_t col = 0; col < m; ++col) {

                typename Matrix_t::coord_tuple_t new_entry(row, col,
                    *vitr.first * ((col + 1) + *citr.first * m));
                coords_new.push_back(new_entry);
            }
        }
    }

    Matrix_t expected_A;
    expected_A.Attach(coords_new);

    if (!static_cast<bool>(expected_A == sketch_A))
        BOOST_FAIL("Result of colwise application not as expected");


/*
    //////////////////////////////////////////////////////////////////////////
    //[> Row wise application <]

    //[> 1. Create the sketching matrix <]
    Dummy_t<DistMatrixType, DistMatrixType> Sparse_r(m, m_s, context);
    row_idx.clear(); row_val.clear();
    row_idx = Sparse_r.getRowIdx();
    row_val = Sparse_r.getRowValues();

    //// PI^T generated by random number gen
    sketch_size = row_val.size();
    mpi_vector_t cols_r(sketch_size);
    mpi_vector_t rows_r(sketch_size);
    mpi_vector_t vals_r(sketch_size);

    for(size_t i = 0; i < sketch_size; ++i) {
        cols_r.SetElement(i, row_idx[i]);
        rows_r.SetElement(i, i);
        vals_r.SetElement(i, row_val[i]);
    }

    DistMatrixType pi_sketch_r(m, m_s, rows_r, cols_r, vals_r);

    //[> 2. Create space for the sketched matrix <]
    DistMatrixType sketch_A_r(n, m_s, zero, zero, zero);

    ////[> 3. Apply the transform <]
    Sparse_r.apply(A, sketch_A_r, skylark::sketch::rowwise_tag());

    //[> 4. Build structure to compare <]
    DistMatrixType expected_AR = PSpGEMM<PTDD>(A, pi_sketch_r);

    if (!static_cast<bool>(expected_AR == sketch_A_r))
        BOOST_FAIL("Result of rowwise application not as expected");
*/
    return 0;
}
