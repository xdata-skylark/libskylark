#include <skylark.hpp>

/*******************************************/
namespace skyml = skylark::ml;
/*******************************************/

int main(int argc, char* argv[]) {
    El::Environment env(argc, argv);
    try {
        const El::Unsigned NROW = El::Input("-d","number of rows/cols", 16);
        const El::Unsigned NCOL = NROW;
        const El::Unsigned M = El::Input("-m","sketching dimension", 8);
        const El::Unsigned ML = El::Input("-l", "ml param", 4);
        const El::Unsigned MR = El::Input("-r", "mr param", 4);
        const El::Unsigned K = El::Input("-k", "k param", 2);
        const double EPS = El::Input("-e", "epsilon param", 0.12);
        const bool sym = El::Input("-s","Apply Symmetric only", false);
        El::ProcessInput();

        El::DistMatrix<double> A(NROW, NCOL);
        El::Uniform(A, NROW, NCOL);

        El::DistMatrix<double> Z(A.Height(), M);
        skyml::low_rank_t<double> transformer (K, EPS, M, ML, MR);

        if (sym) {
            // Apply for sym
            skyml::low_rank_sym_t<double> ret = transformer.apply_symmetric(A);
            El::DistMatrix<double, El::CIRC, El::CIRC> ZU(ret.ZU);
            El::DistMatrix<double, El::CIRC, El::CIRC> D(ret.D);

            El::mpi::Barrier();
            if (El::mpi::Rank() == 0) {
                El::Output("Printing after apply_symmetric:");
                El::Print(ZU, "ZU: ");
                El::Print(D, "D: ");
            }
        } else {
            El::Unsigned nprocs = El::mpi::Size(El::mpi::COMM_WORLD);
            // Apply PSD
            transformer.apply_PSD(A);
            El::DistMatrix<double, El::CIRC, El::CIRC> a(A);
            if (El::mpi::Rank() == 0) {
                El::Print(a, "After apply_PSD: ");
            }
        }
    } catch(std::exception& e) { El::ReportException(e); }

    return EXIT_SUCCESS;
}
