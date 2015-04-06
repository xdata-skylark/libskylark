#ifndef SKYLARK_SKETCH_TRANSFORM_DATA_HPP
#define SKYLARK_SKETCH_TRANSFORM_DATA_HPP

#ifndef SKYLARK_SKETCH_HPP
#error "Include top-level sketch.hpp instead of including individuals headers"
#endif

#include <vector>

#include "boost/foreach.hpp"
#include "boost/property_tree/ptree.hpp"

namespace skylark { namespace sketch {

/**
 * Base class for all sketch transform parameter objects.
 */
struct sketch_params_t {

};

template<typename InputMatrixType, typename OutputMatrixType>
class sketch_transform_t;

/**
 * Base class for all sketch tranform data.
 */
struct sketch_transform_data_t {


    /**
     * Get a concrete sketch transform based on the data
     */
    virtual sketch_transform_t<boost::any, boost::any> *get_transform() = 0;

    /**
     *  Serializes a sketch to a property_tree
     */
    virtual
    boost::property_tree::ptree to_ptree() const = 0;

    virtual ~sketch_transform_data_t() {

    }

    static
    sketch_transform_data_t* from_ptree(const boost::property_tree::ptree& pt);

    std::string get_type() {
        return _type;
    }

protected:

    sketch_transform_data_t (int N, int S, const base::context_t& context,
        const std::string type)
        : _N(N), _S(S), _creation_context(context), _type(type) {

    }

    /**
     * Add common components (ones in the bsae class) to the property_tree.
     */
    void add_common(boost::property_tree::ptree& pt) const {
        pt.put("skylark_object_type", "sketch");
        pt.put("sketch_type", _type);
        pt.put("skylark_version", VERSION);
        pt.put("N", _N);
        pt.put("S", _S);
        pt.put_child("creation_context", _creation_context.to_ptree());
    }


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
};

} } /** namespace skylark::sketch */

#endif /** SKYLARK_SKETCH_TRANSFORM_DATA_HPP */
