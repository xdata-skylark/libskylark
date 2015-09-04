#include <iostream>
#include <unordered_map>
#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#define SKYLARK_NO_ANY
#include <skylark.hpp>

namespace bpo = boost::program_options;

template<typename VertexType>
struct simple_unweighted_graph_t {

    typedef VertexType vertex_type;
    typedef typename std::vector<vertex_type>::const_iterator iterator_type;
    typedef typename std::unordered_map<vertex_type,
                                        std::vector<vertex_type> >::const_iterator
    vertex_iterator_type;

    simple_unweighted_graph_t(const std::string &gf);
    simple_unweighted_graph_t(const hdfsFS &fs, const std::string &gf);

    size_t num_vertices() const { return _nodemap.size(); }
    size_t num_edges() const { return _num_edges; }

    size_t degree(const vertex_type &vertex) const {
        return _nodemap.at(vertex).size();
    }

    iterator_type adjanct_begin(const vertex_type &vertex) const {
        return _nodemap.at(vertex).begin();
    }

    iterator_type adjanct_end(const vertex_type &vertex) const {
        return _nodemap.at(vertex).end();
    }

    vertex_iterator_type vertex_begin() const {
        return _nodemap.begin();
    }

    vertex_iterator_type vertex_end() const {
        return _nodemap.end();
    }

    template<typename T>
    void adjancy_matrix(skylark::base::sparse_matrix_t<T> &A,
        std::vector<vertex_type> &indexmap) const;

private:
    std::unordered_map<vertex_type, std::vector<vertex_type> > _nodemap;
    size_t _num_edges;
};

template<typename VertexType>
simple_unweighted_graph_t<VertexType>::simple_unweighted_graph_t(const std::string &gf) {

    std::ifstream in(gf);
    std::string line, token;
    vertex_type u, v;

    std::unordered_set<std::pair<vertex_type, vertex_type>,
                       skylark::utility::pair_hasher_t> added;

    _num_edges = 0;
    while(true) {
        getline(in, line);
        if (in.eof())
            break;
        if (line[0] == '#')
            continue;

        std::istringstream tokenstream(line);
        tokenstream >> u;
        tokenstream >> v;

        if (u == v)
            continue;

        if (added.count(std::make_pair(u, v)) > 0)
            continue;

        added.insert(std::make_pair(u, v));
        added.insert(std::make_pair(v, u));
        _num_edges += 2;

        _nodemap[u].push_back(v);
        _nodemap[v].push_back(u);
    }

    in.close();
}

template<typename VertexType>
simple_unweighted_graph_t<VertexType>::
simple_unweighted_graph_t(const hdfsFS &fs, const std::string &gf) {

    skylark::utility::hdfs_line_streamer_t in(fs, gf, 1000);
    std::string line, token;
    vertex_type u, v;

    std::unordered_set<std::pair<vertex_type, vertex_type>,
                       skylark::utility::pair_hasher_t> added;

    _num_edges = 0;
    while(true) {
        in.getline(line);
        if (in.eof())
            break;
        if (line[0] == '#')
            continue;

        std::istringstream tokenstream(line);
        tokenstream >> u;
        tokenstream >> v;

        if (u == v)
            continue;

        if (added.count(std::make_pair(u, v)) > 0)
            continue;

        added.insert(std::make_pair(u, v));
        added.insert(std::make_pair(v, u));
        _num_edges += 2;

        _nodemap[u].push_back(v);
        _nodemap[v].push_back(u);
    }

    in.close();
}

template<typename VertexType>
template<typename T>
void simple_unweighted_graph_t<VertexType>::
adjancy_matrix(skylark::base::sparse_matrix_t<T> &A,
    std::vector<vertex_type> &indexmap) const {

    typedef typename skylark::base::sparse_matrix_t<T>::index_type index_type;
    typedef typename skylark::base::sparse_matrix_t<T>::value_type value_type;

    index_type n = num_vertices();
    index_type nnz = num_edges();

    // Allocate space for the matrix
    index_type *colptr = new index_type[n + 1];
    index_type *rowind = new index_type[nnz];
    value_type *values = new value_type[nnz];

    // Create index & node map and initial population of colptr
    std::unordered_map<vertex_type, index_type> nodemap;
    indexmap.resize(n);
    n = 0;
    index_type c = 0;
    for (vertex_iterator_type vit = vertex_begin();
         vit != vertex_end(); vit++) {
        nodemap[vit->first] = n;
        indexmap[n] = vit->first;
        colptr[n] = c;
        c += degree(vit->first);
        n++;
    }
    colptr[n] = c;

    for(index_type j = 0; j < n; j++) {
        index_type i = 0;
        for(iterator_type it = adjanct_begin(indexmap[j]);
            it != adjanct_end(indexmap[j]); it++) {
            values[colptr[j] + i] = 1.0;
            rowind[colptr[j] + i] = nodemap[*it];
            i++;
        }
    }

    A.attach(colptr, rowind, values, nnz, n, n, true);
}

int seed, k, powerits, port;
int oversampling_ratio, oversampling_additive;
std::string graphfile, indexfile, prefix, hdfs;
bool use_single, skipqr, directory;

template<typename T>
void execute() {

    typedef std::string vertex_type;
    typedef simple_unweighted_graph_t<vertex_type> graph_type;

    skylark::base::context_t context(seed);
    graph_type *G;

    boost::mpi::communicator world;
    int rank = world.rank();

    boost::mpi::timer timer;

    if (rank == 0) {
        std::cout << "Reading the graph... ";
        std::cout.flush();
        timer.restart();
    }

    if (!hdfs.empty()) {
#       if SKYLARK_HAVE_LIBHDFS

        hdfsFS fs;
        if (rank == 0)
            fs = hdfsConnect(hdfs.c_str(), port);

        G = new graph_type(fs, graphfile);

#       else

        SKYLARK_THROW_EXCEPTION(skylark::base::io_exception() <<
            skylark::base::error_msg("Install libhdfs for HDFS support!"));

#       endif
    } else
        G = new graph_type(graphfile);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";


    if (rank == 0) {
        std::cout << "Computing embeddings... ";
        std::cout.flush();
        timer.restart();
    }

    std::vector<vertex_type> indexmap;
    El::Matrix<T> X;

    skylark::ml::approximate_ase_params_t params;
    params.skip_qr = skipqr;
    params.num_iterations = powerits;
    params.oversampling_ratio = oversampling_ratio;
    params.oversampling_additive = oversampling_additive;

    skylark::ml::ApproximateASE(*G, k, indexmap, X, context, params);

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    if (rank == 0) {
        std::cout << "Writing results... ";
        std::cout.flush();
        timer.restart();
    }

    El::Write(X, prefix + ".vec", El::ASCII);

    std::ofstream of(prefix + ".index.txt");
    for(size_t i = 0; i < indexmap.size(); i++)
        of << i << "\t" << indexmap[i] << std::endl;
    of.close();

    if (rank == 0)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";
}

int main(int argc, char** argv) {

    El::Initialize(argc, argv);

    // Parse options
    bpo::options_description
        desc("Options");
    desc.add_options()
        ("help,h", "produce a help message")
        ("graphfile,g",
            bpo::value<std::string>(&graphfile),
            "File holding the graph. REQUIRED.")
        //("directory,d", "Whether inputfile is a directory of files whose"
        //    " concatination is the input.")
        ("seed,s",
            bpo::value<int>(&seed)->default_value(38734),
            "Seed for random number generation. OPTIONAL.")
        ("hdfs",
            bpo::value<std::string>(&hdfs)->default_value(""),
            "If not empty, will assume file is in an HDFS. "
            "Parameter is filesystem name.")
        ("port",
            bpo::value<int>(&port)->default_value(0),
            "For HDFS: port to use.")
        ("rank,k",
            bpo::value<int>(&k)->default_value(10),
            "Target rank. OPTIONAL.")
        ("powerits,i",
            bpo::value<int>(&powerits)->default_value(2),
            "Number of power iterations. OPTIONAL.")
        ("skipqr", "Whether to skip QR in each iteration. Higher than one power"
            " iterations is not recommended in this mode.")
        ("ratio,r",
            bpo::value<int>(&oversampling_ratio)->default_value(2),
            "Ratio of oversampling of rank. OPTIONAL.")
        ("additive,a",
            bpo::value<int>(&oversampling_additive)->default_value(0),
            "Additive factor for oversampling of rank. OPTIONAL.")
        ("single", "Whether to use single precision instead of double.")
        ("prefix",
            bpo::value<std::string>(&prefix)->default_value("out"),
            "Prefix for output files (prefix.vec.txt and prefix.index.txt)."
            "  OPTIONAL.");

    bpo::positional_options_description positional;
    positional.add("graphfile", 1);

    bpo::variables_map vm;
    try {
        bpo::store(bpo::command_line_parser(argc, argv)
            .options(desc).positional(positional).run(), vm);

        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }

        use_single = vm.count("single");
        skipqr = vm.count("skipqr");
        //directory = vm.count("directory");

        if (!vm.count("graphfile")) {
            std::cout << "Input graph-file is required." << std::endl;
            return -1;
        }


        bpo::notify(vm);
    } catch(bpo::error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    }


    SKYLARK_BEGIN_TRY()

        if (use_single)
            execute<float>();
        else
            execute<double>();

    SKYLARK_END_TRY() SKYLARK_CATCH_AND_PRINT()

    return 0;
}
