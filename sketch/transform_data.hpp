#ifndef SKYLARK_TRANSFORM_DATA_HPP
#define SKYLARK_TRANSFORM_DATA_HPP

#include <vector>
#include "context.hpp"
#include "../utility/exception.hpp"
#include "../utility/simple_json_parser.hpp"

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
     *  @param[in] json_filename
     *  @param[in] context
     */
    transform_data_t (const std::string& json,
                      context_t& context) : context(context), _version("0.1") {

        utility::simple_json_parser_t parser(json);

        // overwrite/set context to draw correct random samples
        context = context_t(json, context.comm);

        std::vector<int> dims;
        parser.get_vector("sketch.size", dims);
        N = dims[0]; S = dims[1];

        _stream_start = context.get_counter();
    }

    /**
     *  Serializes a sketch to a string.
     *  @param[out] dump containing serialized JSON object
     */
    void dump_json(std::string &dump) const {
        boost::property_tree::ptree sk;

        sk.put("version", _version);
        sk.put("sketch.name", _name);

        boost::property_tree::ptree size;
        boost::property_tree::ptree size_n, size_s;
        size_n.put("", N);
        size_s.put("", S);
        size.push_back(std::make_pair("", size_n));
        size.push_back(std::make_pair("", size_s));
        sk.add_child("sketch.size", size);

        sk.put("sketch.context.counter", _stream_start);
        context.dump_json(sk);

        std::stringstream ss;
        write_json(ss, sk);
        dump = ss.str();
    }

protected:
    int N; /**< Input dimension  */
    int S; /**< Output dimension  */
    skylark::sketch::context_t& context; /**< Context for this sketch */

    std::string _name; /**< distribution name */

private:
    const std::string _version;
    size_t _stream_start; /**< Remember where the random stream started */

};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_TRANSFORM_DATA_HPP */
