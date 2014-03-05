#ifndef SKYLARK_TRANSFORM_DATA_HPP
#define SKYLARK_TRANSFORM_DATA_HPP

#include <vector>
#include "context.hpp"
#include "../utility/exception.hpp"
#include "../utility/simple_json_parser.hpp"

#include "boost/foreach.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

namespace skylark { namespace sketch {

struct transform_data_t {

    transform_data_t (int N, int S, skylark::sketch::context_t& context,
                      const std::string name = "")
        : N(N), S(S), context(context), _name(name), _version("0.1"),
        _stream_start(context.get_counter())
    {}

    /**
     *  Load a serialized sketch from a file.
     *  @param[in] property tree for this sketch
     *  @param[in] context
     */
    transform_data_t (const boost::property_tree::ptree& json,
                      context_t& context) : context(context), _version("0.1") {

        // overwrite/set context to draw correct random samples
        context = context_t(json, context.comm);

        std::vector<int> dims;
        BOOST_FOREACH(const boost::property_tree::ptree::value_type &v,
                      json.get_child("sketch.size")) {

            std::istringstream i(v.second.data());
            int x;
            if (!(i >> x)) dims.push_back(0);
            dims.push_back(x);
        }
        N = dims[0]; S = dims[1];

        _stream_start = context.get_counter();
    }

    /**
     *  Load an array of serialized sketches.
     *  @param[in] filename of the JSON file.
     */
    static void load(const std::string &filename) {

        //TODO
    }

    /**
     *  Load an array of serialized sketches.
     *  @param[in] sketches stream access
     */
    static void load(const std::istream &sketches) {

        //TODO
    }

    friend std::istream& operator>>(std::istream &in, transform_data_t &data);

    /**
     *  Serializes a sketch to a string.
     *  @param[out] dump containing serialized JSON object
     */
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk, const transform_data_t &data);

protected:
    int N; /**< Input dimension  */
    int S; /**< Output dimension  */
    skylark::sketch::context_t& context; /**< Context for this sketch */

    std::string _name; /**< sketch name */

private:
    const std::string _version;
    size_t _stream_start; /**< Remember where the random stream started */
};

std::istream& operator>>(std::istream &in, transform_data_t &data) {

    transform_data_t::load(in);
    return in;
}

boost::property_tree::ptree& operator<<(boost::property_tree::ptree &sk,
                                        const transform_data_t &data) {

    sk.put("sketch.name", data._name);
    sk.put("sketch.version", data._version);

    boost::property_tree::ptree size;
    boost::property_tree::ptree size_n, size_s;
    size_n.put("", data.N);
    size_s.put("", data.S);
    size.push_back(std::make_pair("", size_n));
    size.push_back(std::make_pair("", size_s));
    sk.add_child("sketch.size", size);

    sk.put("sketch.context.counter", data._stream_start);
    sk << data.context;

    return sk;
}

} } /** namespace skylark::sketch */

#endif /** SKYLARK_TRANSFORM_DATA_HPP */
