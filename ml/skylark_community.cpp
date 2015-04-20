#include <iostream>
#include <unordered_map>
#include <El.hpp>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#define SKYLARK_NO_ANY
#include <skylark.hpp>

namespace bmpi =  boost::mpi;
namespace bpo = boost::program_options;
namespace skybase = skylark::base;
namespace skysketch =  skylark::sketch;
namespace skynla = skylark::nla;
namespace skyalg = skylark::algorithms;
namespace skyml = skylark::ml;
namespace skyutil = skylark::utility;


template<typename VertexType>
struct simple_unweighted_graph_t {

    typedef VertexType vertex_type;
    typedef typename std::vector<vertex_type>::const_iterator iterator_type;
    typedef typename std::unordered_map<vertex_type,
                                        std::vector<vertex_type> >::const_iterator
    vertex_iterator_type;

    simple_unweighted_graph_t(const std::string &gf, bool quiet);

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

private:
    std::unordered_map<vertex_type, std::vector<vertex_type> > _nodemap;
    size_t _num_edges;
};

template<typename VertexType>
simple_unweighted_graph_t<VertexType>::simple_unweighted_graph_t(const std::string &gf, bool quiet) {

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

    if (!quiet)
        std::cout << "Finished reading... ";
    in.close();
}

double gamma_, alpha, epsilon;
bool recursive, interactive, quiet;
std::string graphfile, indexfile;
std::vector<std::string> seedss;


template<typename VertexType>
void execute() {
    typedef VertexType vertex_type;

    boost::mpi::timer timer;
    std::unordered_set<vertex_type> seeds;

    if (!quiet) {
        std::cout << "Reading the adjacency matrix... " << std::endl;
        std::cout.flush();
    }
    timer.restart();
    simple_unweighted_graph_t<vertex_type> G(graphfile, quiet);
    if (!quiet)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    bool use_index = !indexfile.empty();
    std::unordered_map<vertex_type, std::string> id_to_name_map;
    std::unordered_map<std::string, vertex_type> name_to_id_map;
    if (use_index) {
        if (!quiet) {
            std::cout << "Reading index files... ";
            std::cout.flush();
        }
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
            std::string name;
            tokenstream >> name;
            vertex_type node;
            tokenstream >> node;

            id_to_name_map[node] = name;
            name_to_id_map[name] = node;
        }

        in.close();

        if (!quiet)
            std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                      << " sec\n";
    }

    do {
        if (interactive) {
            if (!quiet)
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
                    seeds.insert(name_to_id_map[seed]);
                    c++;
                    if (c == 200)
                        exit(-1);
                }
            } else {
                vertex_type seed;
                int c = 0;
                while(strs >> seed) {
                    seeds.insert(seed);
                    c++;
                    if (c == 200)
                        exit(-1);
                }
            }
        } else {
            for(auto it = seedss.begin(); it != seedss.end(); it++)
                if (use_index)
                    seeds.insert(name_to_id_map[*it]);
                else {
                    std::stringstream its(*it);
                    vertex_type seed;
                    its >> seed;
                    seeds.insert(seed);
                }
        }


        timer.restart();
        std::unordered_set<vertex_type> cluster;
        double cond = skyml::FindLocalCluster(G, seeds, cluster,
            alpha, gamma_, epsilon, 4, recursive);
        if (!quiet) {
            std::cout <<"Analysis complete! Took "
                      << boost::format("%.2e") % timer.elapsed() << " sec\n";
            std::cout << "Cluster found:" << std::endl;
        }
        for (auto it = cluster.begin(); it != cluster.end(); it++)
            if (use_index)
                std::cout << id_to_name_map[*it] << std::endl;
            else
                std::cout << *it << " ";
        if (!use_index)
            std::cout << std::endl;
        if (!quiet)
            std::cout << "Conductivity = " << cond << std::endl;
    } while (interactive);
}

template<typename VertexType>
void execute_all() {
    typedef VertexType vertex_type;

    boost::mpi::timer timer;
    std::unordered_set<vertex_type> seeds;

    if (!quiet) {
        std::cout << "Reading the adjacency matrix... " << std::endl;
        std::cout.flush();
    }
    timer.restart();
    simple_unweighted_graph_t<vertex_type> G(graphfile, quiet);
    if (!quiet)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";

    bool use_index = !indexfile.empty();
    std::unordered_map<vertex_type, std::string> id_to_name_map;
    std::unordered_map<std::string, vertex_type> name_to_id_map;
    if (use_index) {
        if (!quiet) {
            std::cout << "Reading index files... ";
            std::cout.flush();
        }
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
            std::string name;
            tokenstream >> name;
            vertex_type node;
            tokenstream >> node;

            id_to_name_map[node] = name;
            name_to_id_map[name] = node;
        }

        in.close();

        if (!quiet)
        std::cout <<"took " << boost::format("%.2e") % timer.elapsed()
                  << " sec\n";
    }

    for(auto it = G.vertex_begin(); it != G.vertex_end(); it++) {
        vertex_type seed = it->first;
        std::unordered_set<vertex_type> seeds;
        seeds.insert(seed);
        std::unordered_set<vertex_type> cluster;
        double cond = skyml::FindLocalCluster(G, seeds, cluster,
            alpha, gamma_, epsilon, 4, recursive);

        if (!quiet)
            std::cout << "Seed: " << seed
                      << " Size: " << cluster.size()
                      << " Cond: " << boost::format("%.3f") % cond
                      << " Community: ";
        for (auto it1 = cluster.begin(); it1 != cluster.end(); it1++)
            if (use_index)
                std::cout << id_to_name_map[*it1] << std::endl;
            else
                std::cout << *it1 << " ";
        if (!use_index)
            std::cout << std::endl;
    }
}


int main(int argc, char** argv) {

    El::Initialize(argc, argv);

    bool numeric, doall;

    // Parse options
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
        ("quiet,q", "Whether to run quietly in interactive mode.")
        ("all,a", "Do all vertexs as seed.")
        ("seed,s",
            bpo::value<std::vector<std::string> >(&seedss),
            "Seed node. Use multiple times for multiple seeds. REQUIRED. ")
        ("recursive,r",
            bpo::value<bool>(&recursive)->default_value(true),
            "Whether to try to recursively improve clusters "
            "(use cluster found as a seed)" )
        ("gamma",
            bpo::value<double>(&gamma_)->default_value(5.0),
            "Time to derive the diffusion. As gamma->inf we get closer to ppr.")
        ("alpha",
            bpo::value<double>(&alpha)->default_value(0.85),
            "PPR component parameter. alpha=1 will result in pure heat-kernel.")
        ("epsilon",
            bpo::value<double>(&epsilon)->default_value(0.001),
            "Accuracy parameter for convergence.")
        ("numeric,n",
            "If present, node labels are numeric and the code exploits that.");

    bpo::variables_map vm;
    try {
        bpo::store(bpo::command_line_parser(argc, argv)
            .options(desc).run(), vm);

        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }

        interactive = vm.count("interactive");
        quiet = vm.count("quiet");
        numeric = vm.count("numeric");
        doall = vm.count("all");

        if (!vm.count("graphfile")) {
            std::cout << "Input graph-file is required." << std::endl;
            return -1;
        }

        if (!interactive && !doall && !vm.count("seed")) {
            std::cout << "A seed is required in non-interactive mode."
                      << std::endl;
            return -1;
        }

        if (interactive && doall) {
            std::cout << "All and interactive do not mix."
                      << std::endl;
            return -1;
        }

        bpo::notify(vm);
    } catch(bpo::error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << desc << std::endl;
        return -1;
    }

    SKYLARK_BEGIN_TRY()

    if (doall) {
        if (numeric)
            execute_all<int>();
        else
            execute_all<std::string>();

    } else {
        if (numeric)
            execute<int>();
        else
            execute<std::string>();
    }

    SKYLARK_END_TRY() SKYLARK_CATCH_AND_PRINT()

    return 0;
}
