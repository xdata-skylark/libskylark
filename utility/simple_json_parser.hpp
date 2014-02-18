#ifndef SIMPLE_JSON_PARSER_HPP
#define SIMPLE_JSON_PARSER_HPP

#include <fstream>
#include <sstream>

#include "./exception.hpp"

#include "boost/foreach.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

namespace skylark { namespace utility {

/**
 *  Simple wrapper around boost property tree.
 *  FIXME: construct from stream and maybe refactor.
 */
struct simple_json_parser_t {

    simple_json_parser_t(const std::string filename) {
        parse_file(filename);
    }

    template<typename T>
    void get_value(const std::string &field, T &val) {

        val = _json_tree.get<T>(field);
    }

    template<typename T>
    void get_vector(const std::string &field,
                    std::vector<T> &vals) {

        BOOST_FOREACH(boost::property_tree::ptree::value_type &v,
                      _json_tree.get_child(field)) {

            std::istringstream i(v.second.data());
            T x;
            if (!(i >> x))
                vals.push_back(0);
            vals.push_back(x);
        }
    }

private:

    typedef std::map<std::string,
            std::map<std::string, std::string> > dict_t;

    dict_t _dict;

    boost::property_tree::ptree _json_tree;

    void parse_file(std::string filename) {

        std::ifstream file;
        std::stringstream json;
        file.open(filename.c_str(), std::ios::in);
        if(file) {
            json << file.rdbuf();
            file.close();
        } else {
            SKYLARK_THROW_EXCEPTION (
                utility::skylark_exception()
                    << utility::error_msg("Cannot open JSON file.") );
        }

        try {
            boost::property_tree::read_json(json, _json_tree);
        } catch (std::exception const& e) {
            std::cerr << e.what() << std::endl;
        }
    }
};

}
}

#endif
