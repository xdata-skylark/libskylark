#include <iostream>

#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <skylark.hpp>

template<typename MatrixType>
class feature_map_precond_t :
    public skylark::algorithms::outplace_precond_t<MatrixType, MatrixType> {

public:

    typedef MatrixType matrix_type;

    virtual bool is_id() const { return false; }

    template<typename KernelType, typename InputType>
    feature_map_precond_t(const KernelType &k, double lambda,
        const InputType &X, El::Int s, skylark::base::context_t &context) {
        _lambda = lambda;
        _s = s;

        U.Resize(s, X.Width());
        skylark::sketch::sketch_transform_t<InputType, matrix_type> *S =
            k.template create_rft<InputType, matrix_type>(s,
                skylark::ml::regular_feature_transform_tag(),
                context);
        S->apply(X, U, skylark::sketch::columnwise_tag());
        delete S;

        El::Identity(C, s, s);
        El::Herk(El::LOWER, El::NORMAL, 1.0/_lambda, U, 1.0, C);
        El::HermitianInverse(El::LOWER, C);
    }

    virtual void apply(const matrix_type& B, matrix_type& X) const {
        matrix_type UB(_s, B.Width()), CUB(_s, B.Width());
        El::Gemm(El::NORMAL, El::NORMAL, 1.0, U, B, 0.0, UB);
        El::Hemm(El::LEFT, El::LOWER, 1.0, C, UB, 0.0, CUB);

        X = B;
        El::Gemm(El::ADJOINT, El::NORMAL, -1.0 / (_lambda * _lambda), U, CUB,
            1.0/_lambda, X);
    }

    virtual void apply_adjoint(const matrix_type& B, matrix_type& X) const {
        apply(B, X);
    }

private:
    double _lambda;
    El::Int _s;
    matrix_type U, C;
};

#ifndef SKYLARK_AVOID_BOOST_PO

#include <boost/program_options.hpp>
namespace bpo = boost::program_options;

int parse_program_options(int argc, char* argv[], 
    int &seed, int &s, std::string &fname, std::string &testname,
    double &sigma, double &lambda) {


    bpo::options_description desc("Options");
    desc.add_options()
        ("help,h", "produce a help message")
        ("trainfile",
            bpo::value<std::string>(&fname),
            "Data to train on (libsvm format).")
        ("testfile",
            bpo::value<std::string>(&testname)->default_value(""),
            "Test data (libsvm format).")
        ("seed,s",
            bpo::value<int>(&seed)->default_value(38734),
            "Seed for random number generation. OPTIONAL.")
        ("sigma,x",
            bpo::value<double>(&sigma),
            "Kernel bandwidth.")
        ("lambda,l",
            bpo::value<double>(&lambda),
            "Lambda regularization parameter.")
        ("numfeatures,f",
            bpo::value<int>(&s),
            "Number of random features.");

    bpo::positional_options_description positional;
    positional.add("trainfile", 1);
    positional.add("testfile", 2);

    bpo::variables_map vm;
    try {
        bpo::store(bpo::command_line_parser(argc, argv)
            .options(desc).positional(positional).run(), vm);

        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0]
                      << " [options] input-file-name [test-file-name]"
                      << std::endl;
            std::cout << desc;
            return 0;
        }

        if (!vm.count("testfile")) {
            std::cout << "Input file is required." << std::endl;
            return -1;
        }

        bpo::notify(vm);
    } catch(bpo::error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    }

    return 1000;
}


#else

int parse_program_options(int argc, char* argv[], 
    int &seed, int &s, std::string &fname, std::string &testname,
    double &sigma, double &lambda) {

    seed = 38734;
    sigma = 10.0;
    lambda = 0.01;
    s = 2000;

    int poscount = 0;
    for (int i = 1; i < argc; i += 2) {
        std::string flag = argv[i];
        std::string value = i + 1 < argc ? argv[i+1] : "";

        if (flag == "--seed" || flag == "-s")
                seed = boost::lexical_cast<int>(value);

        if (flag == "--lambda" || flag == "-l")
            lambda = boost::lexical_cast<double>(value);

        if (flag == "--sigma" || flag == "-x")
            sigma = boost::lexical_cast<double>(value);

        if (flag == "--nunmfeatures" || flag == "-f")
            s = boost::lexical_cast<int>(value);

        if (flag == "--trainfile")
            fname = value;
        if (flag == "--testfile")
            testname = value;

        if (flag[0] != '-' && poscount != 0)
            testname = flag;

        if (flag[0] != '-' && poscount == 0) {
            fname = flag;
            poscount++;
        }

        if (flag[0] != '-')
            i--;
    }

    return 1000;
}

#endif

int main(int argc, char* argv[]) {

    El::Initialize(argc, argv);

    int seed, s;
    std::string fname, testname;
    double sigma, lambda;

    int flag = parse_program_options(argc, argv, seed, s,
        fname, testname, sigma, lambda);
    if (flag != 1000)
        return flag;

    skylark::base::context_t context(seed);

    boost::mpi::communicator world;
    int rank = world.rank();

    El::DistMatrix<double> X;
    El::DistMatrix<El::Int> L;

    boost::mpi::timer timer;

    // Load A and Y
    if (rank == 0) {
        std::cout << "Reading the matrix... ";
        std::cout.flush();
        timer.restart();
    }

    skylark::utility::io::ReadLIBSVM(fname, X, L, skylark::base::COLUMNS);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    // Compute kernel matrix
    if (rank == 0) {
        std::cout << "Computing kernel matrix... ";
        std::cout.flush();
        timer.restart();
    }

    skylark::ml::gaussian_t k(X.Height(), sigma);
    El::DistMatrix<double> K;
    skylark::ml::SymmetricGram(El::LOWER, skylark::base::COLUMNS,
        k, X, K);

    // Add regularizer
    El::DistMatrix<double> D;
    El::Ones(D, X.Width(), 1);
    El::UpdateDiagonal(K, lambda, D);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    if (rank == 0) {
        std::cout << "Creating precoditioner... ";
        std::cout.flush();
        timer.restart();
    }

    feature_map_precond_t<El::DistMatrix<double> > P(k, lambda, X, s, context);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    if (rank == 0) {
        std::cout << "Solving... ";
        std::cout.flush();
        timer.restart();
    }

    // Form right hand side
    El::DistMatrix<double> Y;
    std::unordered_map<El::Int, El::Int> coding;
    std::vector<El::Int> rcoding;
    skylark::ml::DummyCoding(El::NORMAL, Y, L, coding, rcoding);

    // Solve
    skylark::algorithms::krylov_iter_params_t cg_params;
    cg_params.iter_lim = 1000;
    cg_params.res_print = 10;
    cg_params.log_level = 2;
    cg_params.am_i_printing = rank == 0;
    cg_params.tolerance = 1e-3;

    El::DistMatrix<double> A;
    El::Zeros(A, X.Width(), Y.Width());
    skylark::algorithms::CG(El::LOWER, K, Y, A, cg_params, P);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    El::Write(A, "A.dat", El::ASCII);

    // Test
    if (!testname.empty()) {
        if (rank == 0) {
            std::cout << "Prediciting... ";
            std::cout.flush();
            timer.restart();
        }

        El::DistMatrix<double> XT;
        El::DistMatrix<El::Int> LT;
        skylark::utility::io::ReadLIBSVM(testname, XT, LT,
            skylark::base::COLUMNS, X.Height());

        El::DistMatrix<double> KT;
        skylark::ml::Gram(skylark::base::COLUMNS, skylark::base::COLUMNS,
            k, X, XT, KT);

        El::DistMatrix<double> YP(Y.Width(), XT.Width());
        El::Gemm(El::ADJOINT, El::NORMAL, 1.0, A, KT, YP);

        El::DistMatrix<El::Int> LP;
        skylark::ml::DummyDecode(El::ADJOINT, YP, LP, rcoding);

        if (rank == 0)
            std::cout << "took " << boost::format("%.2e") % timer.elapsed()
                      << " sec\n";

        int errs = 0;
        if (LT.LocalHeight() > 0)
            for(int i = 0; i < LT.LocalWidth(); i++)
                if (LT.GetLocal(0, i) != LP.GetLocal(0, i))
                    errs++;

        errs = El::mpi::AllReduce(errs, MPI_SUM, LT.DistComm());

        if (rank == 0)
            std::cout << "Error rate: "
                      << boost::format("%.2f") % ((errs * 100.0) / LT.Width())
                      << "%" << std::endl;
    }

    El::Finalize();
    return 0;
}
