#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#if SKYLARK_HAVE_BOOST
#include <boost/test/minimal.hpp>
#endif

#include "../../skylark.hpp"

#include <El.hpp>

namespace test { namespace util {

template < typename InputMatrixType,
           typename OutputMatrixType = InputMatrixType >
struct hash_transform_test_t : public skylark::sketch::hash_transform_t<
    InputMatrixType, OutputMatrixType,
    boost::random::uniform_int_distribution,
    skylark::utility::rademacher_distribution_t > {

    typedef skylark::sketch::hash_transform_t<
        InputMatrixType, OutputMatrixType,
        boost::random::uniform_int_distribution,
        skylark::utility::rademacher_distribution_t >
            hash_t;

    hash_transform_test_t(int N, int S, skylark::base::context_t& context)
        : skylark::sketch::hash_transform_t<InputMatrixType, OutputMatrixType,
          boost::random::uniform_int_distribution,
          skylark::utility::rademacher_distribution_t>(N, S, context)
    {}

    std::vector<size_t> getRowIdx() { return hash_t::row_idx; }
    std::vector<double> getRowValues() { return hash_t::row_value; }
};

template<typename MatrixType>
MatrixType operator-(MatrixType& A, MatrixType& B) {
    MatrixType C;
    El::Copy(A, C);
    El::Axpy(-1.0, B, C);
    return C;
}

template<typename MatrixType>
bool equal(MatrixType& A, MatrixType& B,  double threshold=1.e-4) {
    MatrixType C = A - B;
    double diff_norm = El::Norm(C);
    if (diff_norm < threshold) {
        return true;
    }
    return false;
}

template<typename InputMatrixType,
         typename LeftSingularVectorsMatrixType,
         typename SingularValuesMatrixType,
         typename RightSingularVectorsMatrixType>
bool equal_svd_product(InputMatrixType& A,
    LeftSingularVectorsMatrixType& U,
    SingularValuesMatrixType& S,
    RightSingularVectorsMatrixType& V,
    double threshold=1e-4) {

    El::DistMatrix<double> S_CIRC_CIRC = S;
    std::vector<double> values(S_CIRC_CIRC.Buffer(),
        S_CIRC_CIRC.Buffer() + S_CIRC_CIRC.Height());
    El::Diagonal(S_CIRC_CIRC, values);
    El::DistMatrix<double> S_MC_MR = S_CIRC_CIRC;

    El::DistMatrix<double> A_MC_MR = A;
    El::DistMatrix<double> U_MC_MR = U;
    El::DistMatrix<double> V_MC_MR = V;
    El::DistMatrix<double> US_MC_MR;
    El::DistMatrix<double> USVt_MC_MR;

    US_MC_MR.Resize(U.Height(), S_CIRC_CIRC.Width());
    El::Zero(US_MC_MR);
    USVt_MC_MR.Resize(U.Height(), V.Height());

    El::Zero(USVt_MC_MR);
    El::Gemm(El::NORMAL, El::NORMAL,    1.0, U_MC_MR,
        S_MC_MR, 0.0, US_MC_MR);
    El::Gemm(El::NORMAL, El::TRANSPOSE, 1.0, US_MC_MR,
        V_MC_MR, 0.0, USVt_MC_MR);

    return equal(A_MC_MR, USVt_MC_MR, threshold);
}


#if SKYLARK_HAVE_BOOST

template <typename dense_matrix_t>
void check_equal(const dense_matrix_t& A, const dense_matrix_t& B) {
    double threshold = 1e-7;
    for (int col = 0; col < A.LocalWidth(); col++) {
        for (int row = 0; row < A.LocalHeight(); row++) {
            double diff = fabs(A.GetLocal(row, col) - B.GetLocal(row, col));
            if (diff > threshold) {
                std::cerr << "(" << row << ", " << col << ") diff = "
                          << A.GetLocal(row, col) << " - "
                          << B.GetLocal(row, col) << " = " << diff
                          << std::endl;
                BOOST_FAIL("Matrices differ");
            }
        }
    }
}

void check(El::DistMatrix<double>& A,
    double threshold=1e-4) {
    El::DistMatrix<double> U, V;
    El::DistMatrix<double, El::VR, El::STAR> S_VR_STAR;
    U = A;
    El::SVD(U, U, S_VR_STAR, V);
    bool passed = equal_svd_product(A, U, S_VR_STAR, V, threshold);
    if (!passed) {
        BOOST_FAIL("Failure in [MC, MR] case");
    }

}


template<El::Distribution ColDist>
void check(El::DistMatrix<double, ColDist, El::STAR>& A,
    double threshold=1e-4) {
    El::DistMatrix<double, ColDist, El::STAR> A_CD_STAR, U_CD_STAR;
    El::DistMatrix<double, El::STAR, El::STAR> S_STAR_STAR, V_STAR_STAR;
    A_CD_STAR = A;
    U_CD_STAR = A_CD_STAR;
    El::SVD(U_CD_STAR, U_CD_STAR, S_STAR_STAR, V_STAR_STAR);
    bool passed = equal_svd_product(A_CD_STAR,
        U_CD_STAR, S_STAR_STAR, V_STAR_STAR, threshold);
    if (!passed) {
        BOOST_FAIL("Failure in [VC/VR, *] case");
    }
}

template<El::Distribution RowDist>
void check(El::DistMatrix<double, El::STAR, RowDist>& A,
    double threshold=1e-4) {
    El::DistMatrix<double, RowDist, El::STAR> V_RD_STAR;
    El::DistMatrix<double, El::STAR, El::STAR> S_STAR_STAR, U_STAR_STAR;
    U_STAR_STAR = A;
    El::SVD(U_STAR_STAR, U_STAR_STAR, S_STAR_STAR, V_RD_STAR);
    bool passed = equal_svd_product(A,
        U_STAR_STAR, S_STAR_STAR, V_RD_STAR, threshold);
    if (!passed) {
        BOOST_FAIL("Failure in [*, VC/VR] case");
    }

}

#endif // SKYLARK_HAVE_BOOST

} }

#endif // TEST_UTILS_HPP
