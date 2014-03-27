#ifndef SKYLARK_TRANSFORM_DATA_HPP
#define SKYLARK_TRANSFORM_DATA_HPP

#include <vector>
#include "../base/context.hpp"
#include "../utility/exception.hpp"

#include "boost/foreach.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

namespace skylark { namespace sketch {

//FIXME: Haim wants to call this sketch_transform_data_t
struct transform_data_t {

    transform_data_t (int N, int S, skylark::base::context_t& context,
                      const std::string name = "")
        : _N(N), _S(S), _context(context), _name(name), _version("0.1"),
        _stream_start(context.get_counter())
    {}

    /**
     *  Load a serialized sketch from a file.
     *  @param[in] property tree for this sketch
     *  @param[in] context
     */
    transform_data_t (const boost::property_tree::ptree& json,
                      skylark::base::context_t& context)
        : _context(context), _version("0.1") {

        // overwrite/set context to draw correct random samples
        _context = context_t(json);

        std::vector<int> dims;
        BOOST_FOREACH(const boost::property_tree::ptree::value_type &v,
                      json.get_child("sketch.size")) {

            std::istringstream i(v.second.data());
            int x;
            if (!(i >> x)) dims.push_back(0);
            dims.push_back(x);
        }
        _N = dims[0]; _S = dims[1];

        _name = json.get<std::string>("sketch.name");
        _stream_start = context.get_counter();
    }

    friend std::istream& operator>>(std::istream &in, transform_data_t &data);

    /**
     *  Serializes a sketch to a string.
     *  @param[out] dump containing serialized JSON object
     */
    friend boost::property_tree::ptree& operator<<(
            boost::property_tree::ptree &sk, const transform_data_t &data);

protected:
    int _N; /**< Input dimension  */
    int _S; /**< Output dimension  */
    skylark::base::context_t& _context; /**< Context for this sketch */

    std::string _name; /**< sketch name */

private:
    const std::string _version;
    size_t _stream_start; /**< Remember where the random stream started */
};


boost::property_tree::ptree& operator<<(boost::property_tree::ptree &sk,
                                        const transform_data_t &data) {

    sk.put("sketch.name", data._name);
    sk.put("sketch.version", data._version);

    boost::property_tree::ptree size;
    boost::property_tree::ptree size_n, size_s;
    size_n.put("", data._N);
    size_s.put("", data._S);
    size.push_back(std::make_pair("", size_n));
    size.push_back(std::make_pair("", size_s));
    sk.add_child("sketch.size", size);

    sk.put("sketch.context.counter", data._stream_start);
    sk << data._context;

    return sk;
}

} } /** namespace skylark::sketch */

#endif /** SKYLARK_TRANSFORM_DATA_HPP */
