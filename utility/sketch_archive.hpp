#ifndef SKYLARK_SKETCH_ARCHIVE_HPP
#define SKYLARK_SKETCH_ARCHIVE_HPP

#include <vector>

#include "../base/exception.hpp"

#include "boost/foreach.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

namespace skylark { namespace utility {

/**
 *  Simple archive for sketches using a Boost property tree.
 *
 *  Storage container for json sketch descriptions.
 */
struct sketch_archive_t {

    boost::property_tree::ptree& get(size_t idx) {
        return _serialized_sketches[idx];
    }

    void types(std::vector<std::string> &types) {
        types.insert(_sketch_types.begin(), _sketch_types.end(), types.begin());
    }

    friend sketch_archive_t& operator<<(
            sketch_archive_t &data, const boost::property_tree::ptree &sketch);

    friend std::ostream& operator<<(
            std::ostream &out, const sketch_archive_t &ar);

    friend std::istream& operator>>(std::istream &in, sketch_archive_t &ar);

private:
    std::vector<std::string> _sketch_types;
    std::vector<boost::property_tree::ptree> _serialized_sketches;

};

sketch_archive_t& operator<<(sketch_archive_t &data,
                             const boost::property_tree::ptree &sketch) {

    data._serialized_sketches.push_back(sketch);
    return data;
}

std::ostream& operator<<(std::ostream &out, const sketch_archive_t &ar) {

    // accumulate all sketches and dump
    boost::property_tree::ptree sketches;
    for(size_t i = 0; i < ar._serialized_sketches.size(); ++i)
        sketches.push_back(std::make_pair("", ar._serialized_sketches[i]));

    boost::property_tree::ptree dump;
    dump.add_child("sketches", sketches);
    write_json(out, dump);
    return out;
}

std::istream& operator>>(std::istream &in, sketch_archive_t &ar) {

    boost::property_tree::ptree json_tree;
    try {
        boost::property_tree::read_json(in, json_tree);
    } catch (std::exception const& e) {
        SKYLARK_THROW_EXCEPTION (
            base::io_exception()
                << base::error_msg(e.what()) );
    }

    BOOST_FOREACH(const boost::property_tree::ptree::value_type& child,
                  json_tree.get_child("sketches")) {

        boost::property_tree::ptree sketch = child.second;
        ar._serialized_sketches.push_back(sketch);
        ar._sketch_types.push_back(sketch.get<std::string>("sketch.type"));
    }

    return in;
}

} }

#endif
