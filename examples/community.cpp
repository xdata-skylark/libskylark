#include <iostream>
#include <unordered_map>
#include <elemental.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <skylark.hpp>

namespace bmpi =  boost::mpi;
namespace bpo = boost::program_options;
namespace skybase = skylark::base;
namespace skysketch =  skylark::sketch;
namespace skynla = skylark::nla;
namespace skyalg = skylark::algorithms;
namespace skyml = skylark::ml;
namespace skyutil = skylark::utility;


struct simple_unweighted_graph_t {


    simple_unweighted_graph_t(const std::string &gf);

    ~simple_unweighted_graph_t() {
        delete[] _out;
    }

    int num_vertices() const { return _num_vertices; }
    int num_edges() const { return _num_edges; }
    int degree(int vertex) const { return _nodepairs.at(vertex).first; }
    const int *adjanct(int vertex) const { return _nodepairs.at(vertex).second; }
private:
    typedef std::pair<int, int *> nodepair_t;

    std::unordered_map<int, nodepair_t> _nodepairs;
    int *_out;
    int _num_vertices;
    int _num_edges;
};


simple_unweighted_graph_t::simple_unweighted_graph_t(const std::string &gf) {

    std::ifstream in(gf);
    std::string line, token;

    _num_edges = 0;
    while(!in.eof()) {
        getline(in, line);
        if (line[0] == '#')
            continue;

        std::istringstream tokenstream(line);
        tokenstream >> token;
        int i = atoi(token.c_str());
        tokenstream >> token;
        int j = atoi(token.c_str());

        if (i == j)
            continue;

        _nodepairs[i].first++;
        _nodepairs[j].first++;
        _num_edges += 2;
    }

    _num_vertices = _nodepairs.size();

    std::cout << "Finished first pass. Vertices = " << _num_vertices
              << " Edges = " << _num_edges << std::endl;
    _out = new int[_num_edges];

    // Set pointers and zero degrees.
    int count = 0;
    for(auto it = _nodepairs.begin(); it != _nodepairs.end(); it++) {
        int nodeid = it->first;
        int deg = it->second.first;
        _nodepairs[nodeid] = nodepair_t(0, _out + count);
        count += deg;
    }

    // Second pass
    in.clear();
    in.seekg(0, std::ios::beg);
    while(!in.eof()) {
        getline(in, line);
        if (line[0] == '#')
            continue;

        std::istringstream tokenstream(line);
        tokenstream >> token;
        int i = atoi(token.c_str());
        tokenstream >> token;
        int j = atoi(token.c_str());

        if (i == j)
            continue;

        nodepair_t &npi = _nodepairs[i];
        npi.second[npi.first] = j;
        npi.first++;

        nodepair_t &npj = _nodepairs[j];
        npj.second[npj.first] = i;
        npj.first++;
    }

    std::cout << "Finished reading... ";
    in.close();
}

int main(int argc, char** argv) {

    elem::Initialize(argc, argv);

    boost::mpi::timer timer;

    // Parse options
    double gamma, alpha, epsilon;
    bool recursive;
    std::string graphfile;
    std::vector<int> seeds;
    bpo::options_description
        desc("Options:");
    desc.add_options()
        ("help,h", "produce a help message")
        ("graphfile,g",
            bpo::value<std::string>(&graphfile),
            "File holding the graph. REQUIRED.")
        ("seed,s",
            bpo::value<std::vector<int> >(&seeds),
            "Seed node. Use multiple times for multiple seeds. REQUIRED. ")
        ("recursive,r",
            bpo::value<bool>(&recursive)->default_value(true),
            "Whether to try to recursively improve clusters "
            "(use cluster found as a seed)" )
        ("gamma",
            bpo::value<double>(&gamma)->default_value(5.0),
            "Time to derive the diffusion. As gamma->inf we get closer to ppr.")
        ("alpha",
            bpo::value<double>(&alpha)->default_value(0.85),
            "PPR component parameter. alpha=1 will result in pure heat-kernel.")
        ("epsilon",
            bpo::value<double>(&epsilon)->default_value(0.001),
            "Accuracy parameter for convergence.");

    bpo::variables_map vm;
    try {
        bpo::store(bpo::command_line_parser(argc, argv)
            .options(desc).run(), vm);

        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }

        if (!vm.count("graphfile")) {
            std::cout << "Input graph-file is required." << std::endl;
            return -1;
        }

        if (!vm.count("seed")) {
            std::cout << "At least one seed node is required." << std::endl;
            return -1;
        }

        bpo::notify(vm);
    } catch(bpo::error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    }

    skybase::sparse_matrix_t<double> A;
    std::cout << "Reading the adjacency matrix... " << std::endl;
    std::cout.flush();
    timer.restart();
    simple_unweighted_graph_t G(graphfile);
    std::cout <<"took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    timer.restart();
    std::vector<int> cluster;
    double cond = skyml::FindLocalCluster(G, seeds, cluster,
        alpha, gamma, epsilon, recursive);
    std::cout <<"Analysis complete! Took "
              << boost::format("%.2e") % timer.elapsed() << " sec\n";
    std::cout << "Cluster found:" << std::endl;
    for (auto it = cluster.begin(); it != cluster.end(); it++)
        std::cout << *it << " ";
    std::cout << std::endl;
    std::cout << "Conductivity = " << cond << std::endl;

    return 0;
}
