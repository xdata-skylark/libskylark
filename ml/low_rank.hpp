#ifndef SKYLARK_LRPSD_HPP
#define SKYLARK_LRPSD_HPP

#include <skylark.hpp>

/*******************************************/
namespace skysketch = skylark::sketch;
/*******************************************/

namespace skylark { namespace ml {

/**
  * An object to return
  */
template <typename T>
struct low_rank_sym_t {
    El::DistMatrix<T> ZU;
    El::DistMatrix<T> D;

    low_rank_sym_t(const El::DistMatrix<T>& Z,
            const El::DistMatrix<T>& U, const El::DistMatrix<T>& D) {

        ZU.Resize(Z.Height(), U.Width());
        El::Gemm(El::NORMAL, El::NORMAL, (T)1, Z, U, (T)0, ZU); // Popl ZU
        this->D = D;
    }

    void print() {
        El::Print(ZU, "ZU: ");
        El::Print(D, "D: ");
    }
};

template <typename T>
class low_rank_t {
    public:
        low_rank_t(const El::Unsigned k,
        const double eps, const El::Unsigned m,
        const El::Unsigned ml, const El::Unsigned mr) {
            this->_k = k;
            this->_eps = eps;
            this->_m = m;
            this->_ml = ml;
            this->_mr = mr;
            verify_args();
        }

        void apply_PSD(El::DistMatrix<T>& A) {
            symmetrize(A); // TODO: Verify assignment of sym

            El::DistMatrix<T> Z(A.Height(), _m);
            orthogonal_basis(A, Z); // Populates Z

            El::DistMatrix<T> A_hat(_m, _m);
            approx_mat(A, Z, A_hat); // Populate A_hat

            symmetrize(A_hat);

            // Z(A+At)/2)
            El::DistMatrix<T> ZA_At2(A.Height(), _m);
            El::Gemm(El::NORMAL, El::NORMAL, (T)1, Z, A_hat, (T)0, ZA_At2);

            // Solution returned in A
            El::Gemm(El::NORMAL, El::TRANSPOSE, (T)1, ZA_At2, Z, (T)0, A);
        }

        low_rank_sym_t<T> apply_symmetric(El::DistMatrix<T>& A) {
            symmetrize(A);

            El::DistMatrix<T> Z(A.Height(), _m);
            orthogonal_basis(A, Z); // Populates Z

            El::DistMatrix<T> A_hat(_m, _m);
            approx_mat(A, Z, A_hat); // Populate A_hat

            // A^{\hat}_k
            symmetrize(A_hat);

            // Add eig decomp of top-k
            El::DistMatrix<T> D(_k, _k);
            //El::DistMatrix<T> U(_m, _k);
            El::DistMatrix<T> U;
            El::HermitianEigSubset<T> subset;
            subset.indexSubset = true;
            subset.lowerIndex = 0;
            subset.upperIndex = _k;

            El::HermitianEig(El::UPPER, A_hat, U, El::DESCENDING, subset);
            El::GetSubmatrix(A_hat, El::IR(0, _k), El::IR(0, _k), D);

            // Make U a diagonal matrix
            El::DistMatrix<T> diagU;
            El::Zeros(diagU, _k, _k);
            El::SetDiagonal(diagU, U);

            // Get Z_k
            El::DistMatrix<T> Zk(Z.Height(), _k);
            El::GetSubmatrix(Z, El::IR(0, Z.Height()), El::IR(0, _k), Zk);

            return low_rank_sym_t<T>(Zk, diagU, D);
        }

        /**
         * \param A Symmetric matrix.
         * \param Z The returned orthogonal basis matrix
         * \param m The sketching dimension
         **/
        void orthogonal_basis(const El::DistMatrix<T>& A, El::DistMatrix<T>&Z) {
            // Form AR via sketch so result Q == Z is in Range(AR)
            skylark::base::context_t context(0);
            skysketch::JLT_t<El::DistMatrix<T>, El::DistMatrix<T> >
                sketch_transform(A.Width(), _m, context); // Reduce cols to _m
            sketch_transform.apply(A, Z, skysketch::rowwise_tag());

            // QR
            El::DistMatrix<T> R;
            El::qr::Explicit(Z, R);
            // El::qr::ts::FormQ(Z); // NOTE: Could use -- newer api
        }

        void symmetrize(El::DistMatrix<T>& A) {
            El::DistMatrix<T> Atrans(A.Width(), A.Height()); // transpose(A)
            El::Transpose(A, Atrans);
            El::Axpy(T(1), Atrans, A);
            A *= (T)0.5;
        }

        /**
         * \param A shape is (n, n)
         * \param Z shape is (n, m)
         * \param A_hat shape is (m, m)
         **/
        void approx_mat(El::DistMatrix<T>& A, El::DistMatrix<T>& Z,
                El::DistMatrix<T>& A_hat) {

            const El::Unsigned n = A.Height();
            const El::Unsigned m = Z.Width();

            skylark::base::context_t context(0);
            // Form SlZinv
            skysketch::JLT_t<El::DistMatrix<T>, El::DistMatrix<T> >
                Sl(n, _ml, context);
            El::DistMatrix<T> SlZinv(_ml, m);

            Sl.apply(Z, SlZinv, skysketch::columnwise_tag());
            El::Pseudoinverse(SlZinv); // Inverse -> (m, ml)

            // Form ZtSrinv
            skysketch::JLT_t<El::DistMatrix<T>, El::DistMatrix<T> >
                Sr(n, _mr, context);
            El::DistMatrix<T> ZtSrinv(_m, _mr);
            // Tranpose Z
            El::DistMatrix<T> Zt(Z.Width(), Z.Height());
            El::Transpose(Z, Zt);
            Sr.apply(Zt, ZtSrinv, skysketch::rowwise_tag());
            El::Pseudoinverse(ZtSrinv); // Inverser -> (mr, m)

            // Form SlASr
            El::DistMatrix<T> SlA(_ml, n);
            El::DistMatrix<T> SlASr(_ml, _mr);
            Sl.apply(A, SlA, skysketch::columnwise_tag());
            Sr.apply(SlA, SlASr, skysketch::rowwise_tag());

            // Form SlZinv*SlASr
            El::DistMatrix<T> tmp(_m, _mr);
            El::Gemm(El::NORMAL, El::NORMAL, (T)1, SlZinv, SlASr, (T)0, tmp);
            // Form A_hat
            El::Gemm(El::NORMAL, El::NORMAL, (T)1, tmp, ZtSrinv, (T)0, A_hat);
        }

    private:
        double _eps;
        El::Unsigned _k;
        El::Unsigned _m;
        El::Unsigned _ml;
        El::Unsigned _mr;

        void verify_args() {
            if (_eps <= 0)
                throw std::runtime_error("eps(ilon) must be > 0");
            if (_k < 1)
                throw std::runtime_error("k must be >= 1");
            if (_m <= _k)
                throw std::runtime_error("m must be > k");
        }
};

}} // namespace skylark::ml
#endif // SKYLARK_LRPSD_HPP
