#ifndef SKYLARK_TRANSFORM_DATA_HPP
#define SKYLARK_TRANSFORM_DATA_HPP

#include <vector>
#include "../base/context.hpp"

#include "boost/foreach.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"

namespace skylark { namespace sketch {

//FIXME: Haim wants to call this sketch_transform_data_t
struct transform_data_t {

    transform_data_t (int N, int S, base::context_t* context,
                      const std::string type = "")
        : _N(N), _S(S), _creation_context(context), _type(type),
        _owns_data(false), _version("0.1"),
        _stream_start(context->get_counter())
    {}

    /**
     *  Load a serialized sketch from a file.
     *  @param[in] property tree for this sketch
     */
    transform_data_t (const boost::property_tree::ptree& json)
        : _owns_data(true), _version("0.1") {

        std::cout << "HERE2" << std::endl;
        // create a fresh context for this sketch
        _creation_context = new base::context_t(json);
        _stream_start     = _creation_context->get_counter();

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

    ~transform_data_t() {
        std::cout << "HERE3: " << _type << std::endl;
        if(_owns_data) {
            std::cout << "HERE: " << _type << std::endl;
            delete _creation_context;
        }
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
    base::context_t* _creation_context; /**< Context for this sketch */

    std::string _type; /**< sketch type */

private:
    bool _owns_data;
    const std::string _version;
    size_t _stream_start; /**< Remember where the random stream started */
};


boost::property_tree::ptree& operator<<(boost::property_tree::ptree &sk,
                                        const transform_data_t &data) {

    sk.put("sketch.type", data._type);
    sk.put("sketch.version", data._version);

    boost::property_tree::ptree size;
    boost::property_tree::ptree size_n, size_s;
    size_n.put("", data._N);
    size_s.put("", data._S);
    size.push_back(std::make_pair("", size_n));
    size.push_back(std::make_pair("", size_s));
    sk.add_child("sketch.size", size);

    sk.put("sketch.context.counter", data._stream_start);
    sk << *(data._creation_context);

    return sk;
}

} } /** namespace skylark::sketch */

#endif /** SKYLARK_TRANSFORM_DATA_HPP */
