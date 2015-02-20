#include <iostream>
#include <unordered_map>
#include <El.hpp>
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

    typedef int vertex_type;

    simple_unweighted_graph_t(const std::string &gf);

    ~simple_unweighted_graph_t() {
        delete[] _out;
    }

    size_t num_vertices() const { return _num_vertices; }
    size_t num_edges() const { return _num_edges; }
    size_t degree(vertex_type vertex) const { return _nodepairs.at(vertex).first; }
    const vertex_type *adjanct(vertex_type vertex) const {
        return _nodepairs.at(vertex).second;
    }

private:
    typedef std::pair<size_t, vertex_type *> nodepair_t;

    std::unordered_map<vertex_type, nodepair_t> _nodepairs;
    vertex_type *_out;
    size_t _num_vertices;
    size_t _num_edges;
};


simple_unweighted_graph_t::simple_unweighted_graph_t(const std::string &gf) {

    std::ifstream in(gf);
    std::string line, token;

    _num_edges = 0;
    while(true) {
        getline(in, line);
        if (in.eof())
            break;
        if (line[0] == '#')
            continue;

        std::istringstream tokenstream(line);
        tokenstream >> token;
        vertex_type i = atoi(token.c_str());
        tokenstream >> token;
        vertex_type j = atoi(token.c_str());

        if (i == j)
            continue;

        _nodepairs[i].first++;
        _nodepairs[j].first++;
        _num_edges += 2;
    }

    _num_vertices = _nodepairs.size();

    std::cout << "Finished first pass. Vertices = " << _num_vertices
              << " Edges = " << _num_edges << std::endl;
    _out = new vertex_type[_num_edges];

    // Set pointers and zero degrees.
    size_t count = 0;
    for(auto it = _nodepairs.begin(); it != _nodepairs.end(); it++) {
        vertex_type nodeid = it->first;
        size_t deg = it->second.first;
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
        vertex_type i = atoi(token.c_str());
        tokenstream >> token;
        vertex_type j = atoi(token.c_str());

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

    El::Initialize(argc, argv);

    boost::mpi::timer timer;

    // Parse options
    double gamma, alpha, epsilon;
    bool recursive, interactive;
    std::string graphfile, indexfile;
    std::vector<std::string> seedss;
    std::vector<int> seeds;
    bpo::options_description
        desc("Options:");
    desc.add_options()
        ("help,h", "produce a help message")
        ("graphfile,g",
            bpo::value<std::string>(&graphfile),
            "File holding the graph. REQUIRED.")
        ("indexfile,d",
            bpo::value<std::string>(&indexfile)->default_value(""),
            "Index files mapping node-ids to strings. OPTIONAL.")
        ("interactive,i", "Whether to run in interactive mode.")
        ("seed,s",
            bpo::value<std::vector<std::string> >(&seedss),
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

        interactive = vm.count("interactive");

        if (!vm.count("graphfile")) {
            std::cout << "Input graph-file is required." << std::endl;
            return -1;
        }

        if (!interactive && !vm.count("seed")) {
            std::cout << "A seed is required in non-interactive mode."
                      << std::endl;
            return -1;
        }

        bpo::notify(vm);
    } catch(bpo::error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    }

    std::cout << "Reading the adjacency matrix... " << std::endl;
    std::cout.flush();
    timer.restart();
    simple_unweighted_graph_t G(graphfile);
    std::cout <<"took " << boost::format("%.2e") % timer.elapsed() << " sec\n";

    bool use_index = !indexfile.empty();
    std::unordered_map<int, std::string> id_to_name_map;
    std::unordered_map<std::string, int> name_to_id_map;
    if (use_index) {
        std::cout << "Reading index files... ";
        std::cout.flush();
        timer.restart();

        std::ifstream in(indexfile);
        std::string line, token;

        while(true) {
            getline(in, line);
            if (in.eof())
                break;

            if (line[0] == '#')
                continue;

            std::istringstream tokenstream(line);
            tokenstream >> token;
            std::string name = token;
            tokenstream >> token;
            int node = atoi(token.c_str());

            id_to_name_map[node] = name;
            name_to_id_map[name] = node;
        }

        in.close();

        std::cout <<"took " << boost::format("%.2e") % timer.elapsed() << " sec\n";
    }

    do {
        if (interactive) {
            std::cout << "Please input seeds: ";
            std::string line;
            std::getline(std::cin, line);
            if (line.empty())
                break;

            seeds.clear();
            std::stringstream strs(line);
            if (use_index) {
                std::string seed;
                int c = 0;
                while (strs >> seed) {
                    seeds.push_back(name_to_id_map[seed]);
                    c++;
                    if (c == 200)
                        exit(-1);
                }
            } else {
                int seed;
                int c = 0;
                while(strs >> seed) {
                    seeds.push_back(seed);
                    c++;
                    if (c == 200)
                        exit(-1);
                }
            }
        } else {
            for(auto it = seedss.begin(); it != seedss.end(); it++)
                if (use_index)
                    seeds.push_back(name_to_id_map[*it]);
                else
                    seeds.push_back(atoi(it->c_str()));
        }


        timer.restart();
        std::vector<int> cluster;
        double cond = skyml::FindLocalCluster(G, seeds, cluster,
            alpha, gamma, epsilon, recursive);
        std::cout <<"Analysis complete! Took "
                  << boost::format("%.2e") % timer.elapsed() << " sec\n";
        std::cout << "Cluster found:" << std::endl;
        for (auto it = cluster.begin(); it != cluster.end(); it++)
            if (use_index)
                std::cout << id_to_name_map[*it] << std::endl;
            else
                std::cout << *it << " ";
        if (!use_index)
            std::cout << std::endl;
        std::cout << "Conductivity = " << cond << std::endl;
    } while (interactive);

    return 0;
}
