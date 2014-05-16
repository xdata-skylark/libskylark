#ifndef SKYLARK_SKETCH_TRANSFORM_DATA_HPP
#define SKYLARK_SKETCH_TRANSFORM_DATA_HPP

#include <vector>
#include "../base/context.hpp"

#include "boost/foreach.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

namespace skylark { namespace sketch {

struct sketch_transform_data_t {

    sketch_transform_data_t (int N, int S, const base::context_t& context,
        const std::string type = "")
        : _N(N), _S(S), _creation_context(context), _type(type),
          _version("0.1") { }

    /**
     *  Load a serialized sketch from a file.
     *  @param[in] property tree for this sketch
     */
    sketch_transform_data_t (const boost::property_tree::ptree& json)
        : _creation_context(json), _version("0.1") {

        std::vector<int> dims;
        BOOST_FOREACH(const boost::property_tree::ptree::value_type &v,
                      json.get_child("sketch.size")) {

            std::istringstream i(v.second.data());
            int x;
            if (!(i >> x)) dims.push_back(0);
            dims.push_back(x);
        }
        _N = dims[0]; _S = dims[1];

        _type = json.get<std::string>("sketch.type");
    }

    friend std::istream& operator>>(std::istream &in, 
        sketch_transform_data_t &data);

    /**
     *  Serializes a sketch to a string.
     *  @param[out] dump containing serialized JSON object
     */
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk, 
                const sketch_transform_data_t &data);

protected:
    int _N; /**< Input dimension  */
    int _S; /**< Output dimension  */

    /// Store the context on creation for serialization
    const base::context_t _creation_context;

    std::string _type; /**< sketch type */

    /// random samples should only be drawn here, return context after random
    /// samples have been extracted.
    base::context_t build() {
        return _creation_context;
    }

private:
    const std::string _version;
};

/// serialize the sketch
boost::property_tree::ptree& operator<<(boost::property_tree::ptree &sk,
                                        const sketch_transform_data_t &data) {

    sk.put("sketch.type", data._type);
    sk.put("sketch.version", data._version);

    boost::property_tree::ptree size;
    boost::property_tree::ptree size_n, size_s;
    size_n.put("", data._N);
    size_s.put("", data._S);
    size.push_back(std::make_pair("", size_n));
    size.push_back(std::make_pair("", size_s));
    sk.add_child("sketch.size", size);

    sk << data._creation_context;

    return sk;
}

} } /** namespace skylark::sketch */

#endif /** SKYLARK_SKETCH_TRANSFORM_DATA_HPP */
